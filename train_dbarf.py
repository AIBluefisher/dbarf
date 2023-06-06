import random
import numpy as np

import torch
import torch.utils.data.distributed

from dbarf.geometry.depth import inv2depth
from dbarf.geometry.align_poses import align_ate_c2b_use_a2b
from dbarf.model.dbarf import DBARFModel
from dbarf.projection import Projector
from dbarf.pose_util import Pose, rotation_distance
from dbarf.render_ray import render_rays
from dbarf.render_image import render_single_image
from dbarf.sample_ray import RaySamplerSingleImage
from dbarf.visualization.pose_visualizer import visualize_cameras
from dbarf.visualization.feature_visualizer import *
from utils import img2mse, mse2psnr, img_HWC2CHW, colorize, img2psnr
from train_ibrnet import IBRNetTrainer, synchronize
import dbarf.config as config

# torch.autograd.set_detect_anomaly(True)


class DBARFTrainer(IBRNetTrainer):
    def __init__(self, config) -> None:
        super().__init__(config)
        
        self.state = 'pose_only'

    def build_networks(self):
        self.model = DBARFModel(self.config,
                                load_opt=not self.config.no_load_opt,
                                load_scheduler=not self.config.no_load_scheduler,
                                pretrained=self.config.pretrained)

        # create projector
        self.projector = Projector(device=self.device)

    def setup_optimizer(self):
        super().setup_optimizer()

        self.pose_optimizer = torch.optim.Adam([
            dict(params=self.model.pose_learner.parameters(), lr=args.lrate_pose)
        ])

        self.pose_scheduler = torch.optim.lr_scheduler.StepLR(
            self.pose_optimizer, step_size=args.lrate_decay_pose_steps, gamma=0.5)

    def setup_loss_functions(self):
        super().setup_loss_functions()

    def compose_state_dicts(self) -> None:
        super().compose_state_dicts()
        self.state_dicts['models']['pose_learner'] = self.model.pose_learner
        self.state_dicts['optimizers']['pose_optimizer'] = self.pose_optimizer
        self.state_dicts['schedulers']['pose_scheduler'] = self.pose_scheduler

    def train_iteration(self, data_batch) -> None:
        ######################### 3-stages training #######################
        # ---- (1) Train the pose optimizer with self-supervised loss.<---|
        # |             (10000 iterations)                                |
        # |--> (2) Train ibrnet while fixing the pose optimizer.          |
        # |             (10000 iterations)                                |
        # |--> (3) Jointly train the pose optimizer and ibrnet.           |
        # |             (10000 iterations)                                |
        # |-------------------------->------------------------------------|
        if self.iteration % 10000 == 0 and (self.iteration // 10000) % 2 == 0:
            self.state = self.model.switch_state_machine(state='pose_only')
        elif self.iteration % 10000 == 0 and (self.iteration // 10000) % 2 == 1:
            self.state = self.model.switch_state_machine(state='nerf_only')
        if self.iteration != 0 and self.iteration % 30000 == 0:
            self.state = self.model.switch_state_machine(state='joint')

        images = torch.cat([data_batch['rgb'], data_batch['src_rgbs'].squeeze(0)], dim=0).cuda().permute(0, 3, 1, 2)
        all_feat_maps = self.model.feature_net(images)

        feat_maps = (all_feat_maps[0][1:, :32, ...], None) if args.coarse_only else \
                    (all_feat_maps[0][1:, :32, ...], all_feat_maps[1][1:, ...])

        pose_feats = all_feat_maps[0]

        min_depth, max_depth = data_batch['depth_range'][0][0], data_batch['depth_range'][0][1]

        # Start of core optimization loop
        pred_inv_depths, pred_rel_poses, sfm_loss, fmap = self.model.correct_poses(
            fmaps=pose_feats,
            target_image=data_batch['rgb'].cuda(),
            ref_imgs=data_batch['src_rgbs'].cuda(),
            target_camera=data_batch['camera'],
            ref_cameras=data_batch['src_cameras'],
            min_depth=min_depth,
            max_depth=max_depth,
            scaled_shape=data_batch['scaled_shape'])

        # load training rays
        ray_sampler = RaySamplerSingleImage(data_batch, self.device)
        N_rand = int(1.0 * args.N_rand * args.num_source_views / data_batch['src_rgbs'][0].shape[0])

        ray_batch = ray_sampler.random_sample(N_rand,
                                              sample_mode=args.sample_mode,
                                              center_ratio=args.center_ratio,
                                              )

        # The predicted inverse depth is used as a weak supervision to NeRF.
        self.pred_inv_depth = pred_inv_depths[-1]
        inv_depth_prior = pred_inv_depths[-1].detach().clone()
        inv_depth_prior = inv_depth_prior.reshape(-1, 1)[ray_batch['selected_inds']]

        ret = render_rays(ray_batch=ray_batch,
                          model=self.model,
                          projector=self.projector,
                          feat_maps=feat_maps,
                          N_samples=args.N_samples,
                          inv_uniform=args.inv_uniform,
                          N_importance=args.N_importance,
                          det=args.det,
                          white_bkgd=args.white_bkgd,
                          inv_depth_prior=None, #inv_depth_prior, # TODO(chenyu): enabling the adaptive sampling when well tuned
                          rel_poses=pred_rel_poses[:, -1, :])

        loss_all = 0
        loss_dict = {}

        # rendered_depth = ret['outputs_coarse']['depth']
        # loss_depth = self_sup_depth_loss(inv_depth_prior, rendered_depth, min_depth, max_depth)
        # scalars_to_log['loss/self-sup-depth'] = loss_depth

        # compute loss
        self.optimizer.zero_grad()
        self.pose_optimizer.zero_grad()

        if self.state == 'pose_only' or self.state == 'joint':
            loss_dict['sfm_loss'] = sfm_loss['loss']
            self.scalars_to_log['loss/photometric_loss'] = sfm_loss['metrics']['photometric_loss']
            if 'smoothness_loss' in sfm_loss['metrics']:
                self.scalars_to_log['loss/smoothness_loss'] = sfm_loss['metrics']['smoothness_loss']

        coarse_loss = self.rgb_loss(ret['outputs_coarse'], ray_batch)
        if ret['outputs_fine'] is not None:
            fine_loss = self.rgb_loss(ret['outputs_fine'], ray_batch)

        loss_dict['nerf_loss'] = coarse_loss
        if ret['outputs_fine'] is not None:
            loss_dict['nerf_loss'] += fine_loss
            
        if self.state == 'joint':
            # loss_all += loss_depth.item()
            loss_all += self.model.compose_joint_loss(
                loss_dict['sfm_loss'], loss_dict['nerf_loss'], self.iteration)
        elif self.state == 'pose_only':
            loss_all += loss_dict['sfm_loss']
        else: # nerf_only
            # loss_all += loss_depth.item()
            loss_all += loss_dict['nerf_loss']

        # with torch.autograd.detect_anomaly():
        loss_all.backward()

        if self.state == 'pose_only' or self.state == 'joint':
            self.pose_optimizer.step()
            self.pose_scheduler.step()

        if self.state == 'nerf_only' or self.state == 'joint':
            self.optimizer.step()
            self.scheduler.step()

        if self.config.local_rank == 0 and self.iteration % self.config.n_tensorboard == 0:
            mse_error = img2mse(ret['outputs_coarse']['rgb'], ray_batch['rgb']).item()
            self.scalars_to_log['train/coarse-loss'] = mse_error
            self.scalars_to_log['train/coarse-psnr-training-batch'] = mse2psnr(mse_error)
            self.scalars_to_log['loss/final'] = loss_all.item()
            self.scalars_to_log['loss/rgb_coarse'] = coarse_loss
            if ret['outputs_fine'] is not None:
                mse_error = img2mse(ret['outputs_fine']['rgb'], ray_batch['rgb']).item()
                self.scalars_to_log['train/fine-loss'] = mse_error
                self.scalars_to_log['train/fine-psnr-training-batch'] = mse2psnr(mse_error)
                self.scalars_to_log['loss/rgb_fine'] = fine_loss
            
            self.scalars_to_log['lr/IBRNet'] = self.scheduler.get_last_lr()[0]
            self.scalars_to_log['lr/pose'] = self.pose_scheduler.get_last_lr()[0]
            
            aligned_pred_poses, poses_gt = align_predicted_training_poses(
                pred_rel_poses[:, -1, :], self.train_data, self.train_dataset, self.config.local_rank)
            pose_error = evaluate_camera_alignment(aligned_pred_poses, poses_gt)
            visualize_cameras(self.visdom, step=self.iteration, poses=[aligned_pred_poses, poses_gt], cam_depth=0.1)

            self.scalars_to_log['train/R_error_mean'] = pose_error['R_error_mean']
            self.scalars_to_log['train/t_error_mean'] = pose_error['t_error_mean']
            self.scalars_to_log['train/R_error_med'] = pose_error['R_error_med']
            self.scalars_to_log['train/t_error_med'] = pose_error['t_error_med']

    def validate(self) -> float:
        self.model.switch_to_eval()

        target_image = self.train_data['rgb'].squeeze(0).permute(2, 0, 1)
        pred_inv_depth_gray = self.pred_inv_depth.squeeze(0).detach().cpu()
        pred_inv_depth = self.pred_inv_depth.squeeze(0).squeeze(0)
        pred_depth= inv2depth(pred_inv_depth)
        pred_depth_color = colorize(pred_depth.detach().cpu(), cmap_name='jet', append_cbar=True).permute(2, 0, 1)

        self.writer.add_image('train/target_image', target_image, self.iteration)
        self.writer.add_image('train/pred_inv_depth', pred_inv_depth_gray, self.iteration)
        self.writer.add_image('train/pred_depth-color', pred_depth_color, self.iteration)

        # Logging a random validation view.
        val_data = next(self.val_loader_iterator)
        tmp_ray_sampler = RaySamplerSingleImage(val_data, self.device, render_stride=self.config.render_stride, \
                                                sim3=None)

        H, W = tmp_ray_sampler.H, tmp_ray_sampler.W
        gt_img = tmp_ray_sampler.rgb.reshape(H, W, 3)
        score = log_view_to_tb(
            self.writer, self.iteration, self.config, self.model, tmp_ray_sampler, self.projector,
            gt_img, render_stride=self.config.render_stride, prefix='val/',
            data=val_data, dataset=self.val_dataset)

        # Logging current training view.
        tmp_ray_train_sampler = RaySamplerSingleImage(self.train_data, self.device, render_stride=1)
        H, W = tmp_ray_train_sampler.H, tmp_ray_train_sampler.W
        gt_img = tmp_ray_train_sampler.rgb.reshape(H, W, 3)
        log_view_to_tb(self.writer, self.iteration, self.config, self.model, tmp_ray_train_sampler, self.projector,
                       gt_img, render_stride=1, prefix='train/', data=self.train_data, dataset=self.train_dataset)

        torch.cuda.empty_cache()
        self.model.switch_to_train()

        return score


@torch.no_grad()
def get_predicted_training_poses(pred_poses):
    target_pose = torch.eye(4, device=pred_poses.device, dtype=torch.float).repeat(1, 1, 1)

    # World->camera poses.
    pred_poses = Pose.from_vec(pred_poses) # [n_views, 4, 4]
    pred_poses = torch.cat([target_pose, pred_poses], dim=0)

    # Convert camera poses to camera->world.
    pred_poses = pred_poses.inverse()

    return pred_poses


@torch.no_grad()
def align_predicted_training_poses(pred_poses, data, dataset, device):
    target_pose_gt = data['camera'][..., -16:].reshape(1, 4, 4)
    src_poses_gt = data['src_cameras'][..., -16:].reshape(-1, 4, 4)
    poses_gt = torch.cat([target_pose_gt, src_poses_gt], dim=0).to(device).float()
    
    pred_poses = get_predicted_training_poses(pred_poses)

    aligned_pred_poses = align_ate_c2b_use_a2b(pred_poses, poses_gt)

    return aligned_pred_poses, poses_gt


@torch.no_grad()
def evaluate_camera_alignment(aligned_pred_poses, poses_gt):
    # measure errors in rotation and translation
    R_aligned, t_aligned = aligned_pred_poses.split([3, 1], dim=-1)
    R_gt, t_gt = poses_gt.split([3, 1], dim=-1)
    
    R_error = rotation_distance(R_aligned[..., :3, :3], R_gt[..., :3, :3])
    t_error = (t_aligned - t_gt)[..., 0].norm(dim=-1)
    
    mean_rotation_error = np.rad2deg(R_error.mean().cpu())
    mean_position_error = t_error.mean()
    med_rotation_error = np.rad2deg(R_error.median().cpu())
    med_position_error = t_error.median()
    
    return {'R_error_mean': mean_rotation_error, "t_error_mean": mean_position_error,
            'R_error_med': med_rotation_error, 't_error_med': med_position_error}


@torch.no_grad()
def log_view_to_tb(writer, global_step, args, model, ray_sampler, projector, gt_img,
                   render_stride=1, prefix='', data=None, dataset=None) -> float:
    # with torch.no_grad():
    ray_batch = ray_sampler.get_all()
    if model.feature_net is not None:
        images = torch.cat([data['rgb'], data['src_rgbs'].squeeze(0)], dim=0).cuda().permute(0, 3, 1, 2)
        all_feat_maps = model.feature_net(images)
        # pose_feats = all_feat_maps[2][:, ...]
        feat_maps = (all_feat_maps[0][1:, :32, ...], None) if model.net_fine is None else \
                    (all_feat_maps[0][1:, :32, ...], all_feat_maps[1][1:, ...])
        # feat_maps = model.feature_net(ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2))
    else:
        feat_maps = [None, None]

    pred_inv_depth, pred_rel_poses, _, __ = model.correct_poses(
                            fmaps=all_feat_maps[0],
                            target_image=data['rgb'].cuda(),
                            ref_imgs=data['src_rgbs'].cuda(),
                            target_camera=data['camera'].cuda(),
                            ref_cameras=data['src_cameras'].cuda(),
                            min_depth=data['depth_range'][0][0],
                            max_depth=data['depth_range'][0][1],
                            scaled_shape=data['scaled_shape'])
    inv_depth_prior = pred_inv_depth.reshape(-1, 1).detach().clone()

    if prefix == 'val/':
        pred_inv_depth = pred_inv_depth.squeeze(0).squeeze(0)
        pred_inv_depth = colorize(pred_inv_depth.detach().cpu(), cmap_name='jet', append_cbar=True).permute(2, 0, 1)
        writer.add_image(prefix + 'pred_inv_depth', pred_inv_depth, global_step)
        aligned_pred_poses, poses_gt = align_predicted_training_poses(pred_rel_poses, data, dataset, args.local_rank)
        pose_error = evaluate_camera_alignment(aligned_pred_poses, poses_gt)
        writer.add_scalar('val/R_error_mean', pose_error['R_error_mean'], global_step)
        writer.add_scalar('val/t_error_mean', pose_error['t_error_mean'], global_step)
        writer.add_scalar('val/R_error_med', pose_error['R_error_med'], global_step)
        writer.add_scalar('val/t_error_med', pose_error['t_error_med'], global_step)

    ret = render_single_image(ray_sampler=ray_sampler,
                              ray_batch=ray_batch,
                              model=model,
                              projector=projector,
                              chunk_size=args.chunk_size,
                              N_samples=args.N_samples,
                              inv_uniform=args.inv_uniform,
                              det=True,
                              N_importance=args.N_importance,
                              white_bkgd=args.white_bkgd,
                              render_stride=render_stride,
                              inv_depth_prior=None, #inv_depth_prior, # TODO(chenyu): enabling the adaptive sampling when well tuned
                              feat_maps=feat_maps)

    average_im = ray_sampler.src_rgbs.cpu().mean(dim=(0, 1))

    if args.render_stride != 1:
        gt_img = gt_img[::render_stride, ::render_stride]
        average_im = average_im[::render_stride, ::render_stride]

    rgb_gt = img_HWC2CHW(gt_img)
    average_im = img_HWC2CHW(average_im)

    rgb_pred = img_HWC2CHW(ret['outputs_coarse']['rgb'].detach().cpu())

    h_max = max(rgb_gt.shape[-2], rgb_pred.shape[-2], average_im.shape[-2])
    w_max = max(rgb_gt.shape[-1], rgb_pred.shape[-1], average_im.shape[-1])
    rgb_im = torch.zeros(3, h_max, 3*w_max)
    rgb_im[:, :average_im.shape[-2], :average_im.shape[-1]] = average_im
    rgb_im[:, :rgb_gt.shape[-2], w_max:w_max+rgb_gt.shape[-1]] = rgb_gt
    rgb_im[:, :rgb_pred.shape[-2], 2*w_max:2*w_max+rgb_pred.shape[-1]] = rgb_pred

    depth_im = ret['outputs_coarse']['depth'].detach().cpu()
    acc_map = torch.sum(ret['outputs_coarse']['weights'], dim=-1).detach().cpu()

    if ret['outputs_fine'] is None:
        depth_im = img_HWC2CHW(colorize(depth_im, cmap_name='jet', append_cbar=True))
        acc_map = img_HWC2CHW(colorize(acc_map, range=(0., 1.), cmap_name='jet', append_cbar=False))
    else:
        rgb_fine = img_HWC2CHW(ret['outputs_fine']['rgb'].detach().cpu())
        rgb_fine_ = torch.zeros(3, h_max, w_max)
        rgb_fine_[:, :rgb_fine.shape[-2], :rgb_fine.shape[-1]] = rgb_fine
        rgb_im = torch.cat((rgb_im, rgb_fine_), dim=-1)
        depth_im = torch.cat((depth_im, ret['outputs_fine']['depth'].detach().cpu()), dim=-1)
        depth_im = img_HWC2CHW(colorize(depth_im, cmap_name='jet', append_cbar=True))
        acc_map = torch.cat((acc_map, torch.sum(ret['outputs_fine']['weights'], dim=-1).detach().cpu()), dim=-1)
        acc_map = img_HWC2CHW(colorize(acc_map, range=(0., 1.), cmap_name='jet', append_cbar=False))

    # write the pred/gt rgb images and depths
    writer.add_image(prefix + 'rgb_gt-coarse-fine', rgb_im, global_step)
    writer.add_image(prefix + 'depth_gt-coarse-fine', depth_im, global_step)
    writer.add_image(prefix + 'acc-coarse-fine', acc_map, global_step)

    # plot_feature_map(writer, global_step, ray_sampler, feat_maps, prefix)

    # write scalar
    pred_rgb = ret['outputs_fine']['rgb'] if ret['outputs_fine'] is not None else ret['outputs_coarse']['rgb']
    psnr_curr_img = img2psnr(pred_rgb.detach().cpu(), gt_img)
    writer.add_scalar(prefix + 'psnr_image', psnr_curr_img, global_step)

    return psnr_curr_img


def train(args):
    device = "cuda:{}".format(args.local_rank)

    trainer = DBARFTrainer(args)
    trainer.train()


if __name__ == '__main__':
    parser = config.config_parser()
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Configuration for distributed training.
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
        print(f'[INFO] Train in distributed mode')

    train(args)
