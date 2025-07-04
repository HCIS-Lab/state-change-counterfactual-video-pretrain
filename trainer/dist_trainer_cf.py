# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from tqdm.auto import tqdm
import torch.distributed as dist
from datetime import datetime
from base import Multi_BaseTrainer_dist
from model.loss import sim_matrix

from utils import inf_loop

class AllGather_multi(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, n_gpu, args):
        output = [torch.empty_like(tensor) for _ in range(args.world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = args.rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None, None,
        )

class Multi_Trainer_dist_CF(Multi_BaseTrainer_dist):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, args, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, writer=None,
                 visualizer=None, tokenizer=None, max_samples_per_epoch=50000, start_epoch=1):
        super().__init__(args, model, loss, metrics, optimizer, config, writer, start_epoch=start_epoch)
        self.config = config
        self.args = args
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            # take the min
            self.len_epoch = min([len(x) for x in data_loader])
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.lr_scheduler = lr_scheduler
        self.visualizer = visualizer
        self.batch_size = self.data_loader[0].batch_size
        self.log_step = int(np.sqrt(self.batch_size))
        self.total_batch_sum = sum([x.batch_size for x in self.data_loader])
        self.max_samples_per_epoch = max_samples_per_epoch
        self.n_gpu = self.args.world_size
        self.allgather = AllGather_multi.apply
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None

    def _eval_metrics(self, output):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output)
            if self.writer is not None:
                self.writer.log_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _adjust_learning_rate(self, optimizer, epoch, args):
        lr = args.learning_rate1
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
        print('[INFO] Learning rate for next epoch is: {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        print('---------------')
        print(epoch)
        self.model.train()
        total_loss = [0] * len(self.data_loader)
        print('learning_rate: ', self.args.learning_rate1)
        total_metrics = np.zeros(len(self.metrics))
        for loader in self.data_loader:
            loader.train_sampler.set_epoch(epoch)
        for batch_idx, data_li in enumerate(zip(*self.data_loader)):
            if (batch_idx + 1) * self.total_batch_sum > self.max_samples_per_epoch:
                break
            for dl_idx, data in enumerate(data_li):
                if 'video_neg' in data.keys():  # w/ negative sampling
                    data['video'] = torch.cat( (data['video'], data['video_neg']), axis = 0)
                    data['narration'] = torch.cat( (data['narration'], data['text_neg_feat']), axis = 0)
                    data['noun_vec'] = torch.cat((data['noun_vec'], data['noun_vec_neg']), axis=0)
                    data['verb_vec'] = torch.cat((data['verb_vec'], data['verb_vec_neg']), axis=0)

                    data['before'] = torch.cat((data['before'], data['neg_before']), axis = 0)
                    data['after'] = torch.cat((data['after'], data['neg_after']), axis = 0)
                    data['CF1'] = torch.cat((data['CF1'], data['neg_cf1']), axis = 0)
                    data['CF2'] = torch.cat((data['CF2'], data['neg_cf2']), axis = 0)
                    data['CF3'] = torch.cat((data['CF3'], data['neg_cf3']), axis = 0)

                data['narration'] = data['narration'].to(self.device)
                data['before'] = data['before'].to(self.device)
                data['after'] = data['after'].to(self.device)
                data['CF1'] = data['CF1'].to(self.device)
                data['CF2'] = data['CF2'].to(self.device)
                data['CF3'] = data['CF3'].to(self.device)
                data['video'] = data['video'].to(self.device)
                n_embeds = data['noun_vec'].to(self.device)
                v_embeds = data['verb_vec'].to(self.device)

                data['narration'].requires_grad = False
                data['before'].requires_grad = False
                data['after'].requires_grad = False
                data['CF1'].requires_grad = False
                data['CF2'].requires_grad = False
                data['CF3'].requires_grad = False
                
                with torch.no_grad():  # Avoid unnecessary gradient tracking
                    narration = self.allgather(data['narration'], self.n_gpu, self.args)
                    before = self.allgather(data['before'], self.n_gpu, self.args)
                    after = self.allgather(data['after'], self.n_gpu, self.args)
                    CF1 = self.allgather(data['CF1'], self.n_gpu, self.args)
                    CF2 = self.allgather(data['CF2'], self.n_gpu, self.args)
                    CF3 = self.allgather(data['CF3'], self.n_gpu, self.args)
                    text_embeds = [narration, before, after, CF1, CF2, CF3]

                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    video_embeds, frame_embeds = self.model(data['video'])
                    num_neg = self.config['data_loader'][0]['args']['neg_param']
                    # if num_neg != False:
                    #     frame_embeds = frame_embeds[:self.batch_size]

                    all_video_embeds = self.allgather(video_embeds, self.n_gpu, self.args)
                    all_frame_embeds = self.allgather(frame_embeds, self.n_gpu, self.args)
                    n_embeds = self.allgather(n_embeds, self.n_gpu, self.args)
                    v_embeds = self.allgather(v_embeds, self.n_gpu, self.args)

                    loss_dict, loss = self.loss(text_embeds, all_video_embeds, \
                                                v_embeds, n_embeds, 
                                                all_frame_embeds, neg_text_embeds)
                loss.backward()
                self.optimizer.step()

                if self.writer is not None and self.args.rank == 0:
                    # self.writer.log_scalar(f'loss_train_{dl_idx}', loss.detach().item())
                    total = int(self.data_loader[dl_idx].n_samples/self.n_gpu)
                    current = batch_idx * self.data_loader[dl_idx].batch_size
                    final_total = (epoch-1) * total + current
                    self.writer.add_scalar(f'Video-text Align_Loss_training/loss_{dl_idx}', loss_dict['align'], final_total)
                    self.writer.add_scalar(f'TCN Loss_training/loss_{dl_idx}', loss_dict['tcn'], final_total)
                total_loss[dl_idx] += loss.detach().item()
                # if batch_idx % self.log_step == 0 and self.args.local_rank == 0:
                if batch_idx % self.log_step == 0 and self.args.rank == 0:
                    self.logger.info('[{}] Train Epoch: {} dl{} {} Loss: {:.6f}'.format(
                        datetime.now().strftime(r'%m%d_%H:%M:%S'),
                        epoch,
                        dl_idx,
                        self._progress(batch_idx, dl_idx),
                        loss.detach().item()))
                    self.logger.info('[{}] Train Epoch: {} dl{} {} Align Loss: {:.6f}'.format(
                        datetime.now().strftime(r'%m%d_%H:%M:%S'),
                        epoch,
                        dl_idx,
                        self._progress(batch_idx, dl_idx),
                        loss_dict['align']))
                    self.logger.info('[{}] Train Epoch: {} dl{} {} TCN Loss: {:.6f}'.format(
                        datetime.now().strftime(r'%m%d_%H:%M:%S'),
                        epoch,
                        dl_idx,
                        self._progress(batch_idx, dl_idx),
                        loss_dict['tcn']))

            if batch_idx == self.len_epoch:
                break

        log = {
            f'loss_{dl_idx}': total_loss[dl_idx] / self.len_epoch for dl_idx in range(len(self.data_loader))
        }

        if self.writer is not None and self.args.rank == 0:
            for dl_idx in range(len(self.data_loader)):
                tl = total_loss[dl_idx] / self.len_epoch
                self.writer.add_scalar(f'Loss_training/loss_total_{dl_idx}', tl, epoch-1)

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            if self.args.rank == 0:
                log.update(val_log)

        self._adjust_learning_rate(self.optimizer, epoch, self.args)

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = [0] * len(self.valid_data_loader)
        total_val_metrics = [np.zeros(len(self.metrics))] * len(self.valid_data_loader)

        gt_arr = {x: [] for x in range(len(self.valid_data_loader))}
        pred_arr = {x: [] for x in range(len(self.valid_data_loader))}
        type_arr = {x: [] for x in range(len(self.valid_data_loader))}

        with torch.no_grad():
            # for validation we switch the nested loop order, because alternate batches not needed...
            # ... and dataloaders can be of different length
            for dl_idx, dl in enumerate(self.valid_data_loader):
                for batch_idx, data in enumerate(tqdm(dl)):
                    data['video'] = data['video'][0]  # remove batch
                    data['text_feats'] = data['text_feats'][0]
                    data['video'] = data['video'].to(self.device)
                    text_embed = data['text_feats'].to(self.device)

                    vid_embed, _ = self.model(data['video'], return_embeds=True)

                    data_gt = data['correct'][0].to(self.device).unsqueeze(0)
                    data_pred = sim_matrix(vid_embed, text_embed)
                    data_type = data['type'][0].to(self.device).unsqueeze(0)

                    data_gt_all = [torch.zeros_like(data_gt) for _ in range(self.n_gpu)]
                    torch.distributed.all_gather(data_gt_all, data_gt)
                    data_gt_all = torch.cat(data_gt_all, dim=0)

                    data_pred_all = [torch.zeros_like(data_pred) for _ in range(self.n_gpu)]
                    torch.distributed.all_gather(data_pred_all, data_pred)
                    data_pred_all = torch.cat(data_pred_all, dim=0)

                    data_type_all = [torch.zeros_like(data_type) for _ in range(self.n_gpu)]
                    torch.distributed.all_gather(data_type_all, data_type)
                    data_type_all = torch.cat(data_type_all, dim=0)

                    gt_arr[dl_idx].append(data_gt_all.cpu())
                    pred_arr[dl_idx].append(data_pred_all.cpu())
                    type_arr[dl_idx].append(data_type_all.cpu())

            if self.writer is not None and self.args.rank == 0:
                for dl_idx in range(len(self.valid_data_loader)):
                    tl = total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx])
                    self.writer.add_scalar(f'Loss_val/loss_total_{dl_idx}', tl, epoch-1)

        for dl_idx in range(len(self.valid_data_loader)):
            nested_metrics = {x: {} for x in range(len(self.valid_data_loader))}

            gt_arr_cat = torch.cat(gt_arr[dl_idx])
            pred_arr_cat = torch.cat(pred_arr[dl_idx])
            type_cat = torch.cat(type_arr[dl_idx])

            for metric in self.metrics:
                metric_name = metric.__name__
                res = metric(pred_arr_cat, gt_arr_cat, type_cat)
                if self.args.rank == 0:
                    self.logger.info(
                        verbose(epoch=epoch, metrics=res, name=self.valid_data_loader[dl_idx].dataset_name))
                nested_metrics[dl_idx][metric_name] = res

                if self.writer is not None and self.args.rank == 0:
                    to_write = format_nested_metrics_for_writer(res, mode=metric_name,
                                                                name=self.valid_data_loader[dl_idx].dataset_name)
                    for key, val in to_write.items():
                        key = key.replace('[', '_').replace(']', '_')
                        self.writer.add_scalar(f'Val_metrics_{dl_idx}/{key}', val, epoch - 1)

        res_dict = {}
        if self.args.rank == 0:
            res_dict = {f'val_loss_{dl_idx}': total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx])
                        for dl_idx in range(len(self.valid_data_loader))}
            res_dict['nested_val_metrics'] = nested_metrics

        return res_dict

    def _progress(self, batch_idx, dl_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader[dl_idx], 'n_samples'):
            current = batch_idx * self.data_loader[dl_idx].batch_size
            total = int(self.data_loader[dl_idx].n_samples / self.n_gpu)
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

def verbose(epoch, metrics, name="TEST"):
    msg = ""
    for key in metrics.keys():
        acc = metrics[key]
        msg += f"{name:s} epoch {epoch}, {key:s}, Acc: {acc:.1f};    "
    print(msg)
    return msg

def format_nested_metrics_for_writer(metrics, mode, name="TEST"):
    res = {}
    for key, val in metrics.items():
        log_name = f"[{mode}]{name}_{key}"
        res[log_name] = val
    return res
