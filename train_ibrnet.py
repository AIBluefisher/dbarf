import os
import time
import numpy as np
import shutil

import torch
import torch.utils.data.distributed
import torch.distributed as dist

from dbarf.base.trainer import BaseTrainer
from dbarf.render_ray import render_rays
from dbarf.render_image import render_single_image
from dbarf.model.ibrnet import IBRNetModel
from dbarf.sample_ray import RaySamplerSingleImage
from dbarf.loss.criterion import MaskedL2ImageLoss
from dbarf.projection import Projector
from utils import img2mse, mse2psnr, img_HWC2CHW, colorize, img2psnr
import dbarf.config as config


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


class IBRNetTrainer(BaseTrainer):
    def __init__(self, config) -> None:
        super().__init__(config)

    def build_networks(self):
        # Create IBRNet model
        self.model = IBRNetModel(self.config,
                                 load_opt=not self.config.no_load_opt,
                                 load_scheduler=not self.config.no_load_scheduler
                                )

        # create projector
        self.projector = Projector(device=self.device)

    def setup_optimizer(self):
        # optimizer and learning rate scheduler
        learnable_params = list(self.model.net_coarse.parameters())
        learnable_params += list(self.model.feature_net.parameters())
        if self.model.net_fine is not None:
            learnable_params += list(self.model.net_fine.parameters())

        if self.model.net_fine is not None:
            self.optimizer = torch.optim.Adam([
                {'params': self.model.net_coarse.parameters()},
                {'params': self.model.net_fine.parameters()},
                {'params': self.model.feature_net.parameters(), 'lr': self.config.lrate_feature}],
                lr=self.config.lrate_mlp)
        else:
            self.optimizer = torch.optim.Adam([
                {'params': self.model.net_coarse.parameters()},
                {'params': self.model.feature_net.parameters(), 'lr': self.config.lrate_feature}],
                lr=self.config.lrate_mlp)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=self.config.lrate_decay_steps,
                                                         gamma=self.config.lrate_decay_factor)

    def setup_loss_functions(self):
        self.rgb_loss = MaskedL2ImageLoss()

    def compose_state_dicts(self) -> None:
        self.state_dicts = {'models': dict(), 'optimizers': dict(), 'schedulers': dict()}
        
        self.state_dicts['models']['net_coarse'] = self.model.net_coarse
        self.state_dicts['models']['feature_net'] = self.model.feature_net
        if self.model.net_fine is not None:
            self.state_dicts['models']['net_fine'] = self.model.net_fine
        
        self.state_dicts['optimizers']['optimizer'] = self.optimizer
        self.state_dicts['schedulers']['scheduler'] = self.scheduler

    def train_iteration(self, data_batch) -> None:
        # load training rays
        ray_sampler = RaySamplerSingleImage(data_batch, self.device)
        N_rand = int(1.0 * self.config.N_rand * self.config.num_source_views \
                    / data_batch['src_rgbs'][0].shape[0])
        ray_batch = ray_sampler.random_sample(N_rand,
                                              sample_mode=self.config.sample_mode,
                                              center_ratio=self.config.center_ratio,
                                              )

        feat_maps = self.model.feature_net(ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2))

        ret = render_rays(ray_batch=ray_batch,
                          model=self.model,
                          projector=self.projector,
                          feat_maps=feat_maps,
                          N_samples=self.config.N_samples,
                          inv_uniform=self.config.inv_uniform,
                          N_importance=self.config.N_importance,
                          det=self.config.det,
                          white_bkgd=self.config.white_bkgd)

        # compute loss
        self.optimizer.zero_grad()
        loss = self.rgb_loss(ret['outputs_coarse'], ray_batch)

        if ret['outputs_fine'] is not None:
            fine_loss = self.rgb_loss(ret['outputs_fine'], ray_batch)
            loss += fine_loss

        loss.backward()
        self.scalars_to_log['loss'] = loss.item()

        self.optimizer.step()
        self.scheduler.step()

        self.scalars_to_log['lr'] = self.scheduler.get_last_lr()[0]

        if self.config.local_rank == 0 and self.iteration % self.config.n_tensorboard == 0:
            mse_error = img2mse(ret['outputs_coarse']['rgb'], ray_batch['rgb']).item()
            self.scalars_to_log['train/coarse-loss'] = mse_error
            self.scalars_to_log['train/coarse-psnr-training-batch'] = mse2psnr(mse_error)
            if ret['outputs_fine'] is not None:
                mse_error = img2mse(ret['outputs_fine']['rgb'], ray_batch['rgb']).item()
                self.scalars_to_log['train/fine-loss'] = mse_error
                self.scalars_to_log['train/fine-psnr-training-batch'] = mse2psnr(mse_error)

    def validate(self) -> float:
        # print('[INFO] Logging a random validation view...')
        self.model.switch_to_eval()

        val_data = next(self.val_loader_iterator)
        tmp_ray_sampler = RaySamplerSingleImage(val_data, self.device, render_stride=args.render_stride)
        H, W = tmp_ray_sampler.H, tmp_ray_sampler.W
        gt_img = tmp_ray_sampler.rgb.reshape(H, W, 3)
        score = log_view_to_tb(self.writer, self.iteration, args, self.model, tmp_ray_sampler, self.projector,
                               gt_img, render_stride=args.render_stride, prefix='val/')
        torch.cuda.empty_cache()

        # print('[INFO] Logging current training view...')
        tmp_ray_train_sampler = RaySamplerSingleImage(self.train_data, self.device, render_stride=1)
        H, W = tmp_ray_train_sampler.H, tmp_ray_train_sampler.W
        gt_img = tmp_ray_train_sampler.rgb.reshape(H, W, 3)
        log_view_to_tb(self.writer, self.iteration, args, self.model,
                       tmp_ray_train_sampler, self.projector,
                       gt_img, render_stride=1, prefix='train/')

        self.model.switch_to_train()
        return score


@torch.no_grad()
def log_view_to_tb(writer, global_step, args, model, ray_sampler, projector, gt_img,
                   render_stride=1, prefix='') -> float:

    # with torch.no_grad():
    ray_batch = ray_sampler.get_all()
    if model.feature_net is not None:
        feat_maps = model.feature_net(ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2))
    else:
        feat_maps = [None, None]
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

    # write scalar
    pred_rgb = ret['outputs_fine']['rgb'] if ret['outputs_fine'] is not None else ret['outputs_coarse']['rgb']
    psnr_curr_img = img2psnr(pred_rgb.detach().cpu(), gt_img)
    writer.add_scalar(prefix + 'psnr_image', psnr_curr_img, global_step)

    return psnr_curr_img


def train(args):
    device = "cuda:{}".format(args.local_rank)

    # # save the args and config files
    # f = os.path.join(out_folder, 'args.txt')
    # with open(f, 'w') as file:
    #     for arg in sorted(vars(args)):
    #         attr = getattr(args, arg)
    #         file.write('{} = {}\n'.format(arg, attr))

    # if args.config is not None:
    #     f = os.path.join(out_folder, 'config.txt')
    #     if not os.path.isfile(f):
    #         shutil.copy(args.config, f)
    trainer = IBRNetTrainer(args)
    trainer.train()


if __name__ == '__main__':
    parser = config.config_parser()
    args = parser.parse_args()

    # Configuration for distributed training.
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
        print(f'[INFO] Train in distributed mode')

    train(args)
