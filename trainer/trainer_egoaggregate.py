# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm
import torch.distributed as dist
from datetime import datetime
import random

from base import Multi_BaseTrainer_dist
from model.counterfactual import sim_matrix
from utils import inf_loop
import wandb

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

class Multi_Trainer_dist_EgoAgg(Multi_BaseTrainer_dist):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, args, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loader=None, agg_data_loader=None, agg_valid_data_loader=None, lr_scheduler=None, len_epoch=None, writer=None,
                 visualizer=None, tokenizer=None, max_samples_per_epoch=50000, additional_losses=None, start_epoch=1):
        super().__init__(args, model, loss, metrics, optimizer, config, writer, start_epoch=start_epoch)
        self.config = config
        self.args = args
        self.data_loader = data_loader
        self.agg_data_loader = agg_data_loader
        self.agg_valid_data_loader = agg_valid_data_loader
        if len_epoch is None:
            # epoch-based training
            # take the min
            self.len_epoch = min([len(x) for x in data_loader])
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.visualizer = visualizer
        # self.val_chunking = True
        self.val_chunking = False
        self.batch_size = self.data_loader[0].batch_size
        self.agg_batch_size = self.agg_data_loader[0].batch_size
        self.log_step = int(np.sqrt(self.batch_size))
        self.total_batch_sum = sum([x.batch_size for x in self.data_loader])
        self.tokenizer = tokenizer
        self.max_samples_per_epoch = max_samples_per_epoch
        self.n_gpu = self.args.world_size
        self.allgather = AllGather_multi.apply
        self.agg_count = 0
        self.additional_losses = additional_losses #Format is [intra-video, intra-text, inter-parent-video, inter-parent-text]
        self.do_hierarchical = self.config['training_methods']['hierarchical']['intra-modal'] or self.config['training_methods']['hierarchical']['inter-modal']
        # self.writer = writer

    def _eval_metrics(self, output):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output)
        return acc_metrics

    def _adjust_learning_rate(self, optimizer, epoch, args):
        lr = args.learning_rate1
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
        print('[INFO] Learning rate for next epoch is: {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    def _train_step(self, data, epoch, batch_idx, dl_idx, hierarchy='child', state=True, cf=True):
        if hierarchy == 'child':
            batch_size = self.batch_size
        elif hierarchy == 'parent':
            batch_size = self.agg_batch_size

        if hierarchy=='child':
            if 'video_neg' in data.keys():  # w/ negative sampling
                data['video'] = torch.cat( (data['video'], data['video_neg']), axis = 0)
                data['noun_vec'] = torch.cat((data['noun_vec'], data['noun_vec_neg']), axis=0)
                data['verb_vec'] = torch.cat((data['verb_vec'], data['verb_vec_neg']), axis=0)
                data['narration'] = torch.cat((data['narration'], data['text_neg_feat']), axis = 0)
                if state:
                    data['before'] = torch.cat((data['before'], data['neg_before']), axis = 0)
                    data['after'] = torch.cat((data['after'], data['neg_after']), axis = 0)
                    data['before'] = data['before'].to(self.device)
                    data['after'] = data['after'].to(self.device)
                    data['before'].requires_grad = False
                    data['after'].requires_grad = False
                    with torch.no_grad():  # Avoid unnecessary gradient tracking
                        before = self.allgather(data['before'], self.n_gpu, self.args)
                        after = self.allgather(data['after'], self.n_gpu, self.args)
                if cf:
                    data['CF1'] = torch.cat((data['CF1'], data['neg_cf1']), axis = 0)
                    data['CF2'] = torch.cat((data['CF2'], data['neg_cf2']), axis = 0)
                    data['CF3'] = torch.cat((data['CF3'], data['neg_cf3']), axis = 0)
                    data['CF1'] = data['CF1'].to(self.device)
                    data['CF2'] = data['CF2'].to(self.device)
                    data['CF3'] = data['CF3'].to(self.device)
                    data['CF1'].requires_grad = False
                    data['CF2'].requires_grad = False
                    data['CF3'].requires_grad = False
                    with torch.no_grad():  # Avoid unnecessary gradient tracking
                        CF1 = self.allgather(data['CF1'], self.n_gpu, self.args)
                        CF2 = self.allgather(data['CF2'], self.n_gpu, self.args)
                        CF3 = self.allgather(data['CF3'], self.n_gpu, self.args)
                        
            
            data['narration'] = data['narration'].to(self.device)
            data['narration'].requires_grad = False
            
            with torch.no_grad():  # Avoid unnecessary gradient tracking
                narration = self.allgather(data['narration'], self.n_gpu, self.args),
            
                if state and not cf:
                    text_embeds = [
                        narration, 
                        before, 
                        after
                        ]
                elif state and cf:
                    text_embeds = [
                        narration, 
                        before, 
                        after, 
                        CF1, CF2, CF3
                        ]
                elif not state and cf:
                    text_embeds = [
                        narration, 
                        CF1, CF2, CF3
                        ]
                else:
                    text_embeds = [narration]

        elif hierarchy == 'parent':    
            
            if cf:
                key_cf = data['CF_key'].to(self.device)
                order_cf = data['CF_order'].to(self.device)
                key_cf = self.allgather(key_cf, self.n_gpu, self.args)
                order_cf = self.allgather(order_cf, self.n_gpu, self.args)

            if 'aggregated_text_feature' in data.keys():
                # data['aggregated_text_feature'] = data['aggregated_text_feature'].to(self.device)
                agg_n_embeds = data['aggregated_noun_vec'].to(self.device)
                agg_v_embeds = data['aggregated_verb_vec'].to(self.device)
            data['summary_feats'] = data['summary_feats'].to(self.device)
            text_embeds = self.allgather(data['summary_feats'], self.n_gpu, self.args)
        
        data['video'] = data['video'].to(self.device)
        n_embeds = data['noun_vec'].to(self.device)
        v_embeds = data['verb_vec'].to(self.device)                    

        self.optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            '''
            Clarification
            -------------
            1. When hierarchy != 'parent': The data dict will not contain 'aggregated_text' parameter
            2. When hierarchy == 'parent': 'text' will always contain summary text and 'aggregated_text' will contain narrations
            '''
            
            if hierarchy == 'parent':
                # text_embeds and video_embeds are the aggregated parent embeddings passed through self.aggregation
                video_stacked_embeds, video_embeds = self.model(data['video'], do_aggregation=True, batch_size=batch_size)

            else:
                video_embeds, frame_embeds = self.model(data['video'])
                frame_embeds = self.allgather(frame_embeds, self.n_gpu, self.args)
            
            n_embeds = self.allgather(n_embeds, self.n_gpu, self.args)
            v_embeds = self.allgather(v_embeds, self.n_gpu, self.args)
            video_embeds = self.allgather(video_embeds, self.n_gpu, self.args)
            

            # Special treatment when we want to aggregate features
            # This part of the code gets the child features based on the selected parent features
            # This has nothing to do with the aggregation strategy
            # However, latest discussion (July end) suggests parent-child matching is not good, only parent-parent and child-child makes sense.
            if hierarchy == 'parent' and self.do_hierarchical:
                # # For handling video aggregation
                # video_stacked_embeds = video_stacked_embeds.view(batch_size, -1, video_stacked_embeds.shape[1])
                # # Do video child feature sampling
                # num_positives = self.config['training_methods']['hierarchical']['num_positives']
                # assert video_stacked_embeds.shape[1] > num_positives
                # pos_indices = random.sample(range(video_stacked_embeds.shape[1]), num_positives)
                # video_clip_embeds = video_stacked_embeds[:, pos_indices, :]

                # video_embeds = torch.mean(video_embeds, dim=1) # Now handled in the forward function of the model. Can be safely removed
                # For handling text aggregation
                # text_stacked_embeds = text_stacked_embeds.view(batch_size, -1, text_stacked_embeds.shape[1])

                agg_n_embeds = agg_n_embeds.contiguous().view(batch_size, -1, agg_n_embeds.shape[1])
                agg_v_embeds = agg_v_embeds.contiguous().view(batch_size, -1, agg_v_embeds.shape[1])
                # Do text child if text aggregation method is used. For summary, we need to invoke a call to model again
                #if self.config['training_methods']['text aggregation']:
                # if True:
                #     num_positives = self.config['training_methods']['hierarchical']['num_positives']
                #     assert text_stacked_embeds.shape[1] > num_positives
                #     assert agg_n_embeds.shape[1] > num_positives
                #     assert agg_v_embeds.shape[1] > num_positives
                #     pos_indices = random.sample(range(text_stacked_embeds.shape[1]), num_positives)
                #     text_clip_embeds = text_stacked_embeds[:, pos_indices, :]
                #     n_clip_embeds = agg_n_embeds[:, pos_indices, :]
                #     v_clip_embeds = agg_v_embeds[:, pos_indices, :]
                # else:
                #     # We need to evaluate on the model again with text aggregation because the current text_embeds only has summary embeddings
                #     text_aggregated_embeds = self.model.module.compute_text(data['aggregated_text'])
                #     text_aggregated_embeds = text_aggregated_embeds.view(batch_size, -1, text_aggregated_embeds.shape[1])
                #     num_positives = self.config['training_methods']['hierarchical']['num_positives']
                #     assert text_aggregated_embeds.shape[1] > num_positives
                #     assert agg_n_embeds.shape[1] > num_positives
                #     assert agg_v_embeds.shape[1] > num_positives
                #     pos_indices = random.sample(range(text_aggregated_embeds.shape[1]), num_positives)
                #     text_clip_embeds = text_aggregated_embeds[:, pos_indices, :]
                #     agg_n_embeds = agg_n_embeds.view(batch_size, -1, agg_n_embeds.shape[1])
                #     agg_v_embeds = agg_v_embeds.view(batch_size, -1, agg_v_embeds.shape[1])
                #     n_clip_embeds = agg_n_embeds[:, pos_indices, :]
                #     v_clip_embeds = agg_v_embeds[:, pos_indices, :]
            

            # if hierarchy == 'parent' and self.do_hierarchical:
            #     video_clip_embeds = self.allgather(video_clip_embeds, self.n_gpu, self.args)
            #     video_clip_embeds = video_clip_embeds.view(-1, video_clip_embeds.shape[-1])
            #     text_clip_embeds = self.allgather(text_clip_embeds, self.n_gpu, self.args)
            #     text_clip_embeds = text_clip_embeds.view(-1, text_clip_embeds.shape[-1])
            #     n_clip_embeds = self.allgather(n_clip_embeds, self.n_gpu, self.args)
            #     n_clip_embeds = n_clip_embeds.view(-1, n_clip_embeds.shape[-1])
            #     v_clip_embeds = self.allgather(v_clip_embeds, self.n_gpu, self.args)
            #     v_clip_embeds = v_clip_embeds.view(-1, v_clip_embeds.shape[-1])

                assert video_embeds.shape[0] == text_embeds.shape[0]
                num_positives_MILNCE = video_embeds.shape[0]

                # if False:#self.config['training_methods']['hierarchical']['intra-modal']:
                #     intra_video_loss = self.additional_losses[0](sim_matrix(video_embeds, video_clip_embeds), num_samples=num_positives_MILNCE)
                #     intra_text_loss = self.additional_losses[1](sim_matrix(text_embeds, text_clip_embeds), num_samples=num_positives_MILNCE)
                #     total_intra_loss = intra_video_loss + intra_text_loss
                # else:
                #     total_intra_loss = None

                # if False:#self.config['training_methods']['hierarchical']['inter-modal']:
                #     inter_parent_video_loss = self.additional_losses[2](sim_matrix(video_embeds, text_clip_embeds), num_samples=num_positives_MILNCE)
                #     inter_parent_text_loss = self.additional_losses[3](sim_matrix(video_clip_embeds, text_embeds), num_samples=num_positives_MILNCE)
                #     total_inter_loss = inter_parent_video_loss + inter_parent_text_loss
                # else:
                #     total_inter_loss = None
            only_sa_no_summary_baseline = False
            # text_embeds = [x.contiguous() for x in text_embeds]
            # video_embeds = video_embeds.contiguous()
            # v_embeds = v_embeds.contiguous()
            # n_embeds = n_embeds.contiguous()

            if hierarchy == 'parent' and not only_sa_no_summary_baseline:
                bsz, cf, d = key_cf.shape
                key_cf = key_cf.contiguous().view(cf,bsz,d)
                order_cf = order_cf.contiguous().view(cf,bsz,d)
                loss_dict, loss = self.loss.forward_summary(text_embeds.contiguous(), video_embeds.contiguous(), key_cf, order_cf, v_embeds.contiguous(), n_embeds.contiguous()) #output1 is text and summary
            else:
                loss_dict, loss = self.loss(text_embeds, video_embeds, \
                                                v_embeds, n_embeds, 
                                                frame_embeds)

            # intra_loss_exists = (hierarchy == 'parent' and self.do_hierarchical and total_intra_loss is not None)
            # inter_loss_exists = (hierarchy == 'parent' and self.do_hierarchical and total_inter_loss is not None)
            # if not intra_loss_exists and not inter_loss_exists:
            #     loss = clip_loss
            # elif intra_loss_exists and not inter_loss_exists:
            #     loss = clip_loss + total_intra_loss
            # elif not intra_loss_exists and inter_loss_exists:
            #     loss = clip_loss + total_inter_loss
            # elif intra_loss_exists and inter_loss_exists:
            #     loss = clip_loss + total_intra_loss + total_inter_loss
            # else:
            #     raise ValueError

        if self.args.rank == 0:
            wandb.log(loss_dict)
        loss = loss.contiguous()
        loss.backward()
        self.optimizer.step()

        # ==================== writer ====================
        if self.writer is not None and self.args.rank == 0 and hierarchy == 'child':
            # self.writer.log_scalar(f'loss_train_{dl_idx}', loss.detach().item())
            total = int(self.data_loader[dl_idx].n_samples/self.n_gpu)
            current = batch_idx * self.data_loader[dl_idx].batch_size
            final_total = (epoch-1) * total + current
            self.writer.add_scalar(f'Loss_training/loss_{dl_idx}', loss.detach().item(), final_total)

        # if batch_idx % self.log_step == 0 and self.args.local_rank == 0:
        if batch_idx % self.log_step == 0 and self.args.rank == 0 and hierarchy == 'child':
            print('[{}] Train Epoch: {} dl{} {} Loss: {:.6f}'.format(
                datetime.now().strftime(r'%m%d_%H:%M:%S'),
                epoch,
                dl_idx,
                self._progress(batch_idx, dl_idx),
                loss.detach().item()))

        if hierarchy == 'parent' and self.args.rank == 0:
            print('[{}] Parent Train Epoch: {} dl{} {} Loss: {:.6f}'.format(
                datetime.now().strftime(r'%m%d_%H:%M:%S'),
                epoch,
                dl_idx,
                'NA',
                loss.detach().item()))
            self.writer.add_scalar(f'Agg_Loss_training/loss_{dl_idx}', loss.detach().item(), self.agg_count)
            self.agg_count+=1

        if hierarchy == 'child' and self.args.rank == 0:
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
        # ==================== writer ====================

        if self.args.rank == 0:
            for param_group in self.optimizer.param_groups:
                curr_lr = param_group['lr']
            #print('Current learning rate is : {}'.format(curr_lr))

        self.optimizer.zero_grad()
        return loss.detach().item()

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

        self.model.train()
        total_loss = [0] * len(self.data_loader)
        total_agg_loss = [0] * len(self.agg_data_loader)
        total_metrics = np.zeros(len(self.metrics))
        for loader in self.data_loader:
            loader.train_sampler.set_epoch(epoch)
        for agg_loader in self.agg_data_loader:
            agg_loader.train_sampler.set_epoch(epoch)
        if len(self.agg_data_loader) != 1:
            print('Unexpected')
            raise ValueError
        self.agg_data_iter = iter(self.agg_data_loader[0])
        for batch_idx, data_li in enumerate(zip(*self.data_loader)):
            if (batch_idx + 1) * self.total_batch_sum > self.max_samples_per_epoch:
                break
            for dl_idx, data in enumerate(data_li):
                catastrophic_forgetting_baseline = False
                if not catastrophic_forgetting_baseline: #In this baseline we only FT for summary
                    # then assume we must tokenize the input, e.g. its a string
                    loss = self._train_step(data, epoch, batch_idx, dl_idx)
                    total_loss[dl_idx] += loss
                    del data

                #Aggregation training step
                if (batch_idx+1) % self.agg_train_freq == 0:
                    try:
                        agg_batch = next(self.agg_data_iter)
                    except StopIteration:
                        self.agg_data_iter = iter(self.agg_data_loader[0])
                        agg_batch = next(self.agg_data_iter)
                    agg_loss = self._train_step(agg_batch, epoch, 0, dl_idx, hierarchy='parent')
                    total_agg_loss[dl_idx] += agg_loss
                    del agg_batch

            if batch_idx == self.len_epoch:
                break

        log = {
            f'loss_{dl_idx}': total_loss[dl_idx] / self.len_epoch for dl_idx in range(len(self.data_loader))
        }

        if self.writer is not None and self.args.rank == 0:
            for dl_idx in range(len(self.data_loader)):
                tl = total_loss[dl_idx] / self.len_epoch
                self.writer.add_scalar(f'Loss_training/loss_total_{dl_idx}', tl, epoch-1)

        if self.writer is not None and self.args.rank == 0:
            for dl_idx in range(len(self.agg_data_loader)):
                tl = total_agg_loss[dl_idx] / self.len_epoch
                self.writer.add_scalar(f'Agg_Loss_training/loss_total_{dl_idx}', tl, epoch-1)

        #self._adjust_learning_rate(self.optimizer, epoch, self.args)

        return log

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