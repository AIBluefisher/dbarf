import os
import math

import torch

from dbarf.model.ibrnet import IBRNetModel
from dbarf.depth_pose_network import DepthPoseNet
from dbarf.loss.photometric_loss import MultiViewPhotometricDecayLoss


class DBARFModel(IBRNetModel):
    def __init__(self, args, load_opt=True, load_scheduler=True, pretrained=True):
        device = torch.device(f'cuda:{args.local_rank}')

        # create pose optimizer.
        self.pose_learner = DepthPoseNet(iters=12, pretrained=pretrained).to(device)
        self.photometric_loss = MultiViewPhotometricDecayLoss()

        super(DBARFModel, self).__init__(args, load_opt, load_scheduler, half_feat_dim=True)

    def to_distributed(self):
        super().to_distributed()

        if self.args.distributed:
            self.pose_learner = torch.nn.parallel.DistributedDataParallel(
                self.pose_learner,
                device_ids=[self.args.local_rank],
                output_device=[self.args.local_rank]
            )

    def correct_poses(self, fmaps, target_image, ref_imgs, target_camera, ref_cameras,
                      min_depth=0.1, max_depth=100, scaled_shape=(378, 504)):
        """
        Args:
            fmaps: [n_views+1, c, h, w]
            target_image: [1, h, w, 3]
            ref_imgs: [1, n_views, h, w, 3]
            target_camera: [1, 34]
            ref_cameras: [1, n_views, 34]
        Return:
            inv_depths: n_iters*[1, 1, h, w] if training else [1, 1, h, w]
            rel_poses: [n_views, n_iters, 6] if training else [n_views, 6]
        """
        target_intrinsics = target_camera[:, 2:18].reshape(-1, 4, 4)[..., :3, :3] # [1, 3, 3]
        ref_intrinsics = ref_cameras.squeeze(0)[:, 2:18].reshape(-1, 4, 4)[..., :3, :3] # [n_views, 3, 3]
        target_image = target_image.permute(0, 3, 1, 2) # [1, 3, h, w]
        ref_imgs = ref_imgs.squeeze(0).permute(0, 3, 1, 2) # [n_views, 3, h, w]

        inv_depths, rel_poses, fmap = self.pose_learner(
            fmaps=None, # fmaps,
            target_image=target_image,
            ref_imgs=ref_imgs,
            target_intrinsics=target_intrinsics,
            ref_intrinsics=ref_intrinsics,
            min_depth=min_depth, max_depth=max_depth,
            scaled_shape=scaled_shape)
        rel_poses = rel_poses.squeeze(0)

        sfm_loss = 0
        if self.pose_learner.training:
            sfm_loss = self.photometric_loss(target_image, ref_imgs, inv_depths, target_intrinsics, ref_intrinsics, rel_poses)

        return inv_depths, rel_poses, sfm_loss, fmap

    def switch_to_eval(self):
        super().switch_to_eval()
        self.pose_learner.eval()

    def switch_to_train(self):
        super().switch_to_train()
        self.pose_learner.train()

    def switch_state_machine(self, state='joint') -> str:
        if state == 'pose_only':
            self._set_pose_learner_state(opt=True)
            self._set_nerf_state(opt=False)
        
        elif state == 'nerf_only':
            self._set_pose_learner_state(opt=False)
            self._set_nerf_state(opt=True)
        
        elif state == 'joint':
            self._set_pose_learner_state(opt=True)
            self._set_nerf_state(opt=True)
        
        else:
            raise NotImplementedError("Not supported state")
        
        return state

    def _set_pose_learner_state(self, opt=True):
        for param in self.pose_learner.parameters():
            param.requires_grad = opt

    def _set_nerf_state(self, opt=True):
        for param in self.net_coarse.parameters():
            param.requires_grad = opt
        
        for param in self.feature_net.parameters():
            param.requires_grad = opt

        if self.net_fine is not None:
            for param in self.net_fine.parameters():
                param.requires_grad = opt
    
    def compose_joint_loss(self, sfm_loss, nerf_loss, step, coefficient=1e-5):
        # The jointly training loss is composed by the convex_combination:
        #   L = a * L1 + (1-a) * L2
        alpha = math.pow(2.0, -coefficient * step)
        loss = alpha * sfm_loss + (1 - alpha) * nerf_loss
        
        return loss
