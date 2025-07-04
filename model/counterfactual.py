# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import pdb

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from base import BaseModel
from model.video_transformer import SpaceTimeTransformer
from model.vit import VisionTransformer
from utils.util import state_dict_data_parallel_fix

class CF(BaseModel):
    def __init__(self,
                 video_params,
                 text_params,
                 aggregation_params=None,
                 projection_dim=768,
                 load_checkpoint=None,
                 projection='minimal',
                 load_temporal_fix='zeros'):
        super().__init__()

        self.video_params = video_params
        self.load_temporal_fix = load_temporal_fix
        # if not text_params['pretrained']:
        #     raise NotImplementedError("Huggingface text models require pretrained init.")


        pretrained = video_params['pretrained']
        if video_params['model'] == "SpaceTimeTransformer":
            num_frames = video_params.get('num_frames', 4)
            time_init = video_params.get('time_init', 'zeros')
            attention_style = video_params.get('attention_style', 'frozen-in-time')
            arch_config = video_params.get('arch_config', 'base_patch16_224')
            drop_rate = video_params.get('drop_rate', 0.)
            attn_drop_rate = video_params.get('attn_drop_rate', 0.)
            vit_init = video_params.get('vit_init', 'imagenet-21k')
            if arch_config == 'base_patch16_224':
                vit_model = torch.load("/nfs/wattrel/data/md0/datasets/state_aware/jx_vit_base_p16_224-80ecf9dd.pth", map_location="cpu")
                # vit_model = torch.load("/N/project/ego4d_vlm/state-aware-video-pretrain/pretrained/jx_vit_base_p16_224-80ecf9dd.pth", map_location="cpu")
                print("pre-trained model found.")
                model = SpaceTimeTransformer(num_frames=num_frames,
                                            drop_rate=drop_rate,
                                            attn_drop_rate=attn_drop_rate,
                                            time_init=time_init,
                                            attention_style=attention_style)
            else:
                raise NotImplementedError

            model.head = nn.Identity()
            model.pre_logits = nn.Identity()
            ftr_dim = model.embed_dim
            if load_checkpoint in ["", None]:
                # vit_checkpoint = vit_model.state_dict()
                # model.load_state_dict(vit_checkpoint, strict=False)
                vit_checkpoint = vit_model
                new_vit_dict = state_dict_data_parallel_fix(vit_checkpoint, model.state_dict())
                strict = False
                model.load_state_dict(new_vit_dict, strict=strict)
                print(f"model loaded succesfully (strict = {strict})")
            self.video_model = model
        else:
            raise NotImplementedError(f"{video_params['model']} not implemented")

        # for backwards compatibility (old models)
        self.video_model.fc = nn.Identity()


        # Project to a common embedding
        if projection == 'minimal':
            vid_proj = nn.Sequential(
                nn.Linear(ftr_dim, projection_dim)
            )
            frame_proj = nn.Sequential(
                nn.Linear(ftr_dim, projection_dim)
            )
        elif projection == '':
            vid_proj = nn.Identity()
        else:
            raise NotImplementedError
        self.vid_proj = vid_proj
        self.frame_proj = frame_proj

        if aggregation_params is not None and aggregation_params['do_aggregation']:
            if aggregation_params['type'] not in ['self-attention', 'average']:
                raise NotImplementedError(f"{aggregation_params['type']} invalid or not implemented")
            # The input is typically of the shape batch x clip_features x embeddings
            # and the expected out is batch x embeddings since we aggregate features
            if aggregation_params['type'] == 'average':
                self.aggregation = self.average
            elif aggregation_params['type'] == 'self-attention':
                # Also define the model here
                self.aggregation = VisionTransformer()

        if load_checkpoint not in ["", None]:
            # checkpoint = torch.load(load_checkpoint)
            #local_rank = int(os.environ['LOCAL_RANK'])  # fixed by qinghong.
            #checkpoint = torch.load(load_checkpoint, map_location='cuda:{}'.format(local_rank))
            checkpoint = torch.load(load_checkpoint, map_location='cpu')
            state_dict = checkpoint['state_dict']
            new_state_dict = state_dict_data_parallel_fix(state_dict, self.state_dict())
            newest_state_dict = state_dict_data_parallel_fix(new_state_dict, self.state_dict())
            newest_state_dict = self._inflate_positional_embeds(newest_state_dict)
            self.load_state_dict(newest_state_dict, strict=False)

    def set_device(self, device):
        self.device = device

    def forward(self, video, video_only=False, return_embeds=True, do_aggregation=False, batch_size=None):
        if do_aggregation and batch_size is None:
            raise NotImplementedError("If do_aggregation is activated, batch_size must be provided.")
        if video_only:
            video_data = video
            video_embeddings, frame_embeddings = self.compute_video(video_data)
            # return video_embeddings, frame_embeddings
            return video_embeddings

        video_data = video
        video_embeddings, frame_embeddings = self.compute_video(video_data)

        if do_aggregation:
            video_parent_embeddings = self.aggregation(video_embeddings, batch_size)

        if return_embeds and not do_aggregation:
            return  video_embeddings, frame_embeddings
        elif return_embeds and do_aggregation:
            return video_embeddings, video_parent_embeddings

    def compute_video_aggregation(self, video_embeddings, batch_size):
        # Needed because we want to do aggregation separately as well
        return self.aggregation(video_embeddings, batch_size)

    def compute_video(self, video_data):
        video_embeddings, frame_embeddings = self.video_model(video_data)
        return self.vid_proj(video_embeddings), self.frame_proj(frame_embeddings)

    def average(self, embeddings, batch_size):
        # Expected embeddings input shape: (batch_size x samples_per_video) x embed_dim and output shape: batch_size x embed_dim
        # We need to find the average along the samples_per_video dimension
        embeddings = embeddings.view(batch_size, -1, embeddings.shape[1])
        return torch.mean(embeddings, dim=1)

    def _inflate_positional_embeds(self, new_state_dict):
        # allow loading of timesformer with fewer num_frames
        curr_keys = list(self.state_dict().keys())
        if 'video_model.temporal_embed' in new_state_dict and 'video_model.temporal_embed' in curr_keys:
            load_temporal_embed = new_state_dict['video_model.temporal_embed']
            load_num_frames = load_temporal_embed.shape[1]
            curr_num_frames = self.video_params['num_frames']
            embed_dim = load_temporal_embed.shape[2]

            if load_num_frames != curr_num_frames:
                if load_num_frames > curr_num_frames:
                    print(f'### loaded {self.video_params["model"]} model has MORE frames than current...'
                          f'### loading weights, filling in the extras via {self.load_temporal_fix}')
                    new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
                else:
                    print(f'### loaded {self.video_params["model"]} model has FEWER frames than current...'
                          f'### loading weights, filling in the extras via {self.load_temporal_fix}')
                    if self.load_temporal_fix == 'zeros':
                        new_temporal_embed = torch.zeros([load_temporal_embed.shape[0], curr_num_frames, embed_dim])
                        new_temporal_embed[:, :load_num_frames] = load_temporal_embed
                    elif self.load_temporal_fix in ['interp', 'bilinear']:
                        # interpolate
                        # unsqueeze so pytorch thinks its an image
                        mode = 'nearest'
                        if self.load_temporal_fix == 'bilinear':
                            mode = 'bilinear'
                        load_temporal_embed = load_temporal_embed.unsqueeze(0)
                        new_temporal_embed = F.interpolate(load_temporal_embed,
                                                           (curr_num_frames, embed_dim), mode=mode, align_corners=True).squeeze(0)
                    else:
                        raise NotImplementedError
                new_state_dict['video_model.temporal_embed'] = new_temporal_embed
        # allow loading with smaller spatial patches. assumes custom border crop, to append the
        # border patches to the input sequence
        if 'video_model.pos_embed' in new_state_dict and 'video_model.pos_embed' in curr_keys:
            load_pos_embed = new_state_dict['video_model.pos_embed']
            load_num_patches = load_pos_embed.shape[1]
            curr_pos_embed = self.state_dict()['video_model.pos_embed']
            if load_num_patches != curr_pos_embed.shape[1]:
                raise NotImplementedError(
                    'Loading models with different spatial resolution / patch number not yet implemented, sorry.')

        return new_state_dict

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

if __name__ == "__main__":
    pass
