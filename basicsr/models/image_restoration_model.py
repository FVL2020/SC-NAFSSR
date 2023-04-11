# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import importlib
import torch
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils.dist_util import get_dist_info

loss_module = importlib.import_module('basicsr.losses.basic_loss')
metric_module = importlib.import_module('basicsr.metrics')

@MODEL_REGISTRY.register()
class ImageRestorationModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageRestorationModel, self).__init__(opt)

        # define network
        self.net_g = build_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        self.fix_flow_iter, self.train_pam_iter = None, None
        if self.is_train:
            self.init_training_settings()
            if self.opt['train'].get('fix_flow', False):
                self.fix_flow_iter = opt['train'].get('fix_flow')
            if self.opt['train'].get('pam_iter', False):
                self.train_pam_iter = opt['train'].get('pam_iter')
        
        self.scale = int(opt['scale'])

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('stereo_consistent_opt'):
            percep_type = train_opt['stereo_consistent_opt'].pop('type')
            cri_sc_cls = getattr(loss_module, percep_type)
            self.cri_sc = cri_sc_cls(
                **train_opt['stereo_consistent_opt']).to(self.device)
        else:
            self.cri_sc = None
        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        # if optim_type == 'Adam':
        #     self.optimizer_g = torch.optim.Adam([{'params': optim_params}],
        #                                         **train_opt['optim_g'])
        # elif optim_type == 'SGD':
        #     self.optimizer_g = torch.optim.SGD(optim_params,
        #                                        **train_opt['optim_g'])
        # elif optim_type == 'AdamW':
        #     self.optimizer_g = torch.optim.AdamW([{'params': optim_params}],
        #                                         **train_opt['optim_g'])
        #     pass
        # else:
        #     raise NotImplementedError(
        #         f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data, is_val=False):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def grids(self):
        b, c, h, w = self.gt.size()
        self.original_size = (b, c, h, w)

        assert b == 1
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)


        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale
        #adaptive step_i, step_j
        num_row = (h - 1) // crop_size_h + 1
        num_col = (w - 1) // crop_size_w + 1

        import math
        step_j = crop_size_w if num_col == 1 else math.ceil((w - crop_size_w) / (num_col - 1) - 1e-8)
        step_i = crop_size_h if num_row == 1 else math.ceil((h - crop_size_h) / (num_row - 1) - 1e-8)

        scale = self.scale
        step_i = step_i//scale*scale
        step_j = step_j//scale*scale

        parts = []
        idxes = []

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size_h >= h:
                i = h - crop_size_h
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + crop_size_w >= w:
                    j = w - crop_size_w
                    last_j = True
                parts.append(self.lq[:, :, i // scale :(i + crop_size_h) // scale, j // scale:(j + crop_size_w) // scale])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        self.idxes = idxes

    def grids_inverse(self):
        preds = torch.zeros(self.original_size)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w))
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)

        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i: i + crop_size_h, j: j + crop_size_w] += self.outs[cnt]
            count_mt[0, 0, i: i + crop_size_h, j: j + crop_size_w] += 1.

        self.output = (preds / count_mt).to(self.device)
        self.lq = self.origin_lq

    def optimize_parameters(self, current_iter, tb_logger):
        if self.fix_flow_iter:
            logger = get_root_logger()
            if current_iter == 1:
                logger.info(f'Fix flow network and feature extractor for {self.fix_flow_iter} iters.')
                for name, param in self.net_g.named_parameters():
                    if 'spynet' in name or 'edvr' in name:
                        param.requires_grad_(False)
            elif current_iter == self.fix_flow_iter:
                logger.warning('Train all the parameters.')
                self.net_g.requires_grad_(True)
        self.optimizer_g.zero_grad()

        if self.train_pam_iter:
            if current_iter == 1:
                logger = get_root_logger()
                logger.info(f'Only train PAM module for {self.train_pam_iter} iters.')
                for name, param in self.net_g.named_parameters():
                    if ('fusion' not in name) and ('patch_embed_ln' not in name):
                        param.requires_grad = False
            elif current_iter == self.train_pam_iter:
                self.train_pam_iter = None
                logger = get_root_logger()
                logger.warning('Train all the parameters.')
                for param in self.net_g.parameters():
                    param.requires_grad = True

        if self.opt['train'].get('mixup', False):
            self.mixup_aug()
        preds, cri_sc = self.net_g(self.lq)
        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = 0.
            for pred in preds:
                l_pix += self.cri_pix(pred, self.gt)

            # print('l pix ... ', l_pix)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # perceptual loss
        if self.cri_perceptual:
            l_percep_l, l_style_l = self.cri_perceptual(self.output[:,:3], self.gt[:,:3])
            l_percep_r, l_style_r = self.cri_perceptual(self.output[:,3:], self.gt[:,3:])

            if l_percep_l is not None:
                l_percep = l_percep_l + l_percep_r
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style_l is not None:
                l_style = l_style_l + l_style_r
                l_total += l_style
                loss_dict['l_style'] = l_style

        # stereo consistent loss
        if self.cri_sc:
            l_sc = self.cri_sc(self.output, self.gt)
            l_total += l_sc
            loss_dict['l_sc'] = l_sc
        else:
            l_sc = cri_sc
            l_total += l_sc
            loss_dict['l_sc'] = l_sc
        l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())
        l_total.backward()
        use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()


        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        # mod_pad_h, mod_pad_w = 0, 0
        # _, _, h, w = self.lq.size()
        # if h % 12 != 0:
        #     mod_pad_h = 12 - h % 12
        # if w % 12 != 0:
        #     mod_pad_w = 12 - w % 12
        # img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'replicate')
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output, _ = self.net_g_ema(self.lq)
                # _, _, h, w = output.size()
                # self.output = output[:, :, 0:h - mod_pad_h * 4, 0:w - mod_pad_w * 4]
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output, _ = self.net_g(self.lq)
                # _, _, h, w = output.size()
                # self.output = output[:, :, 0:h - mod_pad_h * 4, 0:w - mod_pad_w * 4]
            self.net_g.train()
        # if hasattr(self, 'net_g_ema'):
        #     self.net_g_ema.eval()
        #     with torch.no_grad():
        #         n = len(self.lq)
        #         outs = []
        #         m = self.opt['val'].get('max_minibatch', n)
        #         i = 0
        #         while i < n:
        #             j = i + m
        #             if j >= n:
        #                 j = n
        #             pred = self.net_g_ema(self.lq[i:j])
        #             if isinstance(pred, list):
        #                 pred = pred[-1]
        #             outs.append(pred.detach().cpu())
        #             i = j
        #         self.output = torch.cat(outs, dim=0)
        # else:
        #     self.net_g.eval()
        #     with torch.no_grad():
        #         n = len(self.lq)
        #         outs = []
        #         m = self.opt['val'].get('max_minibatch', n)
        #         i = 0
        #         while i < n:
        #             j = i + m
        #             if j >= n:
        #                 j = n
        #             pred = self.net_g(self.lq[i:j])
        #             if isinstance(pred, list):
        #                 pred = pred[-1]
        #             outs.append(pred.detach().cpu())
        #             i = j
        #         self.output = torch.cat(outs, dim=0)
        #     self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
            if 'lpips' in self.opt['val']['metrics'] or 'score' in self.opt['val']['metrics']:
                from basicsr.metrics.ntire_score import NTIRE_score
                score_metric = NTIRE_score(self.opt['val']['metrics']['score']['fast']).cuda()
                self.opt['val']['metrics']['lpips']['calculate_lpips'] = score_metric.cal_lpips
                self.opt['val']['metrics']['score']['calculate_score'] = score_metric

        rank, world_size = get_dist_info()
        if rank == 0:
            pbar = tqdm(total=len(dataloader), unit='image')

        cnt = 0
        
        for idx, val_data in enumerate(dataloader):
            if idx % world_size != rank:
                continue

            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data, is_val=True)
            if self.opt['val'].get('grids', False):
                self.grids()

            self.test()

            if self.opt['val'].get('grids', False):
                self.grids_inverse()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if sr_img.shape[2] == 6:
                    L_img = sr_img[:, :, :3]
                    R_img = sr_img[:, :, 3:]

                    # visual_dir = osp.join('visual_results', dataset_name, self.opt['name'])
                    visual_dir = osp.join(self.opt['path']['visualization'], dataset_name)

                    imwrite(L_img, osp.join(visual_dir, f'{img_name}_L.png'))
                    imwrite(R_img, osp.join(visual_dir, f'{img_name}_R.png'))
                else:
                    if self.opt['is_train']:

                        save_img_path = osp.join(self.opt['path']['visualization'],
                                                 img_name,
                                                 f'{img_name}_{current_iter}.png')

                        save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                                 img_name,
                                                 f'{img_name}_{current_iter}_gt.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}.png')
                        save_gt_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_gt.png')

                    imwrite(sr_img, save_img_path)
                    imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        if 'lpips' in metric_type or 'score' in metric_type:
                            with torch.no_grad():
                            #     self.metric_results[name] += getattr(
                            #     metric_module, metric_type)(visuals['result'], visuals['gt'], lpips_vgg, **opt_).mean().cpu().numpy()
                                self.metric_results[name] += self.opt['val']['metrics'][name][metric_type](visuals['result'], visuals['gt'])\
                                    .mean().detach().cpu().numpy()
                        else:
                            self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)

                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1
            if rank == 0:
                for _ in range(world_size):
                    pbar.update(1)
                    pbar.set_description(f'Test {img_name}')
        if rank == 0:
            pbar.close()

        # current_metric = 0.
        collected_metrics = OrderedDict()
        if with_metrics:
            for metric in self.metric_results.keys():
                collected_metrics[metric] = torch.tensor(self.metric_results[metric]).float().to(self.device)
            collected_metrics['cnt'] = torch.tensor(cnt).float().to(self.device)

            self.collected_metrics = collected_metrics

            keys = []
            metrics = []
            for name, value in self.collected_metrics.items():
                keys.append(name)
                metrics.append(value)
            metrics = torch.stack(metrics, 0)
            torch.distributed.reduce(metrics, dst=0)
            if self.opt['rank'] == 0:
                metrics_dict = {}
                cnt = 0
                for key, metric in zip(keys, metrics):
                    if key == 'cnt':
                        cnt = float(metric)
                        continue
                    metrics_dict[key] = float(metric)

                for key in metrics_dict:
                    metrics_dict[key] /= cnt

                self._log_validation_metric_values(current_iter, dataloader.dataset.opt['name'],
                                                tb_logger, metrics_dict)
        return 0.

    def nondist_validation(self, *args, **kwargs):
        logger = get_root_logger()
        logger.warning('nondist_validation is not implemented. Run dist_validation.')
        self.dist_validation(*args, **kwargs)


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger, metric_dict):
        log_str = f'Validation {dataset_name}, \t'
        for metric, value in metric_dict.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)

        log_dict = OrderedDict()
        # for name, value in loss_dict.items():
        for metric, value in metric_dict.items():
            log_dict[f'm_{metric}'] = value

        self.log_dict = log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
