# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# Copyright 2020 Ross Wightman
# Modified Model definition

import torch
import torch.nn as nn
from functools import partial
import math
import warnings
import torch.nn.functional as F
import numpy as np

from lib.models.vit_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from lib.models.helpers import load_pretrained
from lib.models.vit_utils import DropPath, to_2tuple, trunc_normal_
from lib.models.tfm_model import DiffusionTransformer as OrderTransformer

from lib.models.build import MODEL_REGISTRY
from torch import einsum
from einops import rearrange, reduce, repeat
import clip
import ipdb

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'mvit': _cfg(
        url='https://dl.fbaipublicfiles.com/mvit/mvitv2_models/MViTv2_S_in1k.pyth',
    ),
}

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='divided_space_time'):
        super().__init__()
        self.attention_type = attention_type
        assert(attention_type in ['divided_space_time', 'space_only','joint_space_time'])

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
           dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        ## Temporal Attention Parameters
        if self.attention_type == 'divided_space_time':
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(
              dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.temporal_fc = nn.Linear(dim, dim)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x, B, T, W):
        num_spatial_tokens = (x.size(1) - 1) // T
        H = num_spatial_tokens // W

        if self.attention_type in ['space_only', 'joint_space_time']:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        elif self.attention_type == 'divided_space_time':
            ## Temporal
            xt = x[:,1:,:]
            xt = rearrange(xt, 'b (h w t) m -> (b h w) t m',b=B,h=H,w=W,t=T)
            res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
            res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res_temporal = self.temporal_fc(res_temporal)
            xt = x[:,1:,:] + res_temporal

            ## Spatial
            init_cls_token = x[:,0,:].unsqueeze(1)
            cls_token = init_cls_token.repeat(1, T, 1)
            cls_token = rearrange(cls_token, 'b t m -> (b t) m',b=B,t=T).unsqueeze(1)
            xs = xt
            xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m',b=B,h=H,w=W,t=T)
            xs = torch.cat((cls_token, xs), 1)
            res_spatial = self.drop_path(self.attn(self.norm1(xs)))

            ### Taking care of CLS token
            cls_token = res_spatial[:,0,:]
            cls_token = rearrange(cls_token, '(b t) m -> b t m',b=B,t=T)
            cls_token = torch.mean(cls_token,1,True) ## averaging for every frame
            res_spatial = res_spatial[:,1:,:]
            res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res = res_spatial
            x = xt

            ## Mlp
            x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.proj(x)
        W = x.size(-1)
        x = x.flatten(2).transpose(1, 2)
        return x, T, W


class VisionTransformer(nn.Module):
    """ Vision Transformer
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, num_frames=8, attention_type='divided_space_time',
                 label_emb = '', mlp=0,text_model = '', lp=False,num_seg=0,extra_tr='order', drope=0., cfg=None):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.temp = self.cfg.DEV.TEMP
        self.order_pretrain = cfg.DEV.ORDER_PRETRAIN_ENABLED
        self.order_max_len = cfg.DEV.ORDER_PRETRAIN_MAX_LEN
        self.order_fix_recognition = cfg.DEV.ORDER_FIX_RECOGNITION
        self.order_tfm_layers = cfg.DEV.ORDER_TFM_LAYERS
        self.order_recog_batch = cfg.DEV.ORDER_RECOG_BATCH
        self.softmax = nn.Softmax(dim=1)
        
        ############## Frame-level TimeSformer Encoder ##############
        # raw video frames as input
        self.attention_type = attention_type
        self.depth = depth
        # input projection
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        ## Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)

        ## Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, attention_type=self.attention_type)
            for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)
        ############## Frame-level TimeSformer Encoder ##############

        # Classifier head
        self.mlp = mlp
        self.label = label_emb # text embeddings
        if label_emb != '':  # pretraining
            self.label_emb = torch.load(label_emb)
            # match video emb to language emb
            self.head = nn.Linear(embed_dim, self.label_emb.shape[1])  # project to the dim of emb   
            # order pretraining
            self.order_tfm = OrderTransformer(num_seg=self.order_max_len-1, tfm_layers=self.order_tfm_layers, dropout=self.cfg.MODEL.DROP_E, hidden_size=self.head.weight.shape[0], cfg=self.cfg)
        else:  # finetuning
            if self.cfg.DEV.MATCH_LANG_EMB: # match video emb to language emb
                self.label_emb = torch.load(self.cfg.DEV.TEST_LANG_EMB)
                self.head = nn.Linear(embed_dim, self.label_emb.shape[1])  # project to the dim of emb
                for p in self.head.parameters(): p.requires_grad = False # fix head
            else: # classify video emb into classes
                self.label_emb = False
                self.test_lang_emb = torch.load(self.cfg.DEV.TEST_LANG_EMB)
                self.head = nn.Linear(embed_dim, self.test_lang_emb.shape[1]) 
                for p in self.head.parameters(): p.requires_grad = False
                
                if cfg.TRAIN.DATASET == 'Epickitchens':
                    self.head_n = nn.Linear(self.test_lang_emb.shape[1], 300)
                    self.head_v = nn.Linear(self.test_lang_emb.shape[1], 97)
                else:
                    self.head_cls = nn.Linear(self.test_lang_emb.shape[1], num_classes)  
            self.apply(self._init_weights)
        
        # text encoder
        self.text = text_model  
        if text_model == 'clip_vit_b_16': # create text encoder
            clip_model, _ = clip.load("ViT-B/16", jit=False)
            del clip_model.visual # remove visual branch
            self.text_model = clip_model.float()
            for p in self.text_model.parameters(): p.requires_grad = False

        # clip-level temporal modeling
        if num_seg > 0:
            self.num_seg = num_seg
            self.order_tfm = OrderTransformer(num_seg=self.num_seg, tfm_layers=self.order_tfm_layers, dropout=self.cfg.MODEL.DROP_E, hidden_size=self.head.weight.shape[0], cfg=self.cfg)
                
        # initialization of transformer embeddings
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        ## initialization of temporal attention weights
        if self.depth != 0 and self.attention_type == 'divided_space_time':
            i = 0
            for m in self.blocks.modules():
                m_str = str(m)
                if 'Block' in m_str:
                    if i > 0:
                      nn.init.constant_(m.temporal_fc.weight, 0)
                      nn.init.constant_(m.temporal_fc.bias, 0)
                    i += 1

    def forward(self, x):
        # seperate input video frames and input text (e.g., ASR)
        if len(self.text) > 0 and self.training:
            x, text = x
        batch_size = x.shape[0]

        # divide m*t video frames into m clips
        if self.order_pretrain: # order pretraining
            x = rearrange(x, 'b m c t h w -> (b m) c t h w', m=self.order_max_len)
        elif hasattr(self, 'num_seg') and self.num_seg > 0: # step forecasting
            x = rearrange(x, 'b c (m t) h w -> (b m) c t h w', m=self.num_seg, t=x.shape[2]//self.num_seg)
        
        # TimeSformer encoder
        x = self.forward_features(x) # input: [b, 3, t, h, w] --> output: [b, d]
        
        # projection layers
        if self.cfg.DEV.MATCH_LANG_EMB: # match video emb to language emb (pretraining)
            self.label_emb = self.check_device_norm(self.label_emb, x.device, norm=True)
            x = self.head(x)
            x = x / x.norm(dim=1, keepdim=True)
            video_emb = x
            if hasattr(self, 'num_seg') and self.num_seg > 0: # zero-shot step forecasting
                x = self.order_tfm(video_emb)
                x = x / x.norm(dim=1, keepdim=True)
            x = x @ self.label_emb.t() / self.temp
        else: # classify video emb into classes (finetuning)
            if hasattr(self, 'num_seg') and self.num_seg > 0: # step / action forecasting
                x = self.head(x)
                video_emb = x / x.norm(dim=1, keepdim=True)
                x = self.order_tfm(video_emb)
                x = self.head_cls(x)
            else: # step / action classification
                x = self.head(x)
                x = x / x.norm(dim=1, keepdim=True)
                if hasattr(self, 'head_n'): # EPIC-Kitchen step classification
                    v = self.head_v(x) / self.temp
                    n = self.head_n(x) / self.temp
                    return (v,n)
                else:
                    x = self.head_cls(x) / self.temp
                
        # create pseudo labels using language embeddings during pre-training
        if isinstance(self.label_emb,torch.Tensor) and len(self.text) > 0 and self.training: # order pretraining
            # use teacher model to match video frames&ASR and step descriptions
            teacher_x = self.get_pseudo_labels(x.device, text)

            # recover video embedding of mask token
            pred_video_emb, mask_inds, mse_loss, intermediate_denoise = self.order_tfm(video_emb, is_pretrain=True)

            # get the matching score of mask token
            pred_video_emb = pred_video_emb / pred_video_emb.norm(dim=1, keepdim=True)
            mask_pred = pred_video_emb @ self.label_emb.t() / self.temp 
            
            # get the matching scores of masked-out clip
            masked_teacher_x = self.get_mask_samples(teacher_x, mask_inds=mask_inds) # teacher prediction

            # create teacher target for intermediate denoised tokens
            intermediate_denoise = intermediate_denoise / intermediate_denoise.norm(dim=1, keepdim=True)
            intermediate_pred = intermediate_denoise @ self.label_emb.t() / self.temp 
            intermediate_teacher_x = masked_teacher_x.unsqueeze(0).expand(self.order_tfm.level_batch, -1, -1).reshape(-1, masked_teacher_x.size(-1))

            # organize the batch
            rand_inds = torch.randperm(x.shape[0], device=x.device)[:batch_size*self.order_recog_batch] # get a subset of batch to avoid OOM
            x = x[rand_inds]
            teacher_x = teacher_x[rand_inds]

            # sampled time levels from diffusion model
            x = torch.cat((x, intermediate_pred), dim=0) 
            teacher_x = torch.cat((teacher_x, intermediate_teacher_x), dim=0) 
            return x, teacher_x, mse_loss
                
        # testing
        if not self.training:
            x = self.softmax(x)
        
        return x

    def get_mask_samples(self, all_samples, mask_inds):
        all_samples = rearrange(all_samples, '(b m) c -> b m c', m=self.order_max_len, b=all_samples.shape[0]//self.order_max_len)
        mask_samples = all_samples[torch.arange(all_samples.shape[0], device=all_samples.device), mask_inds, :]
        return mask_samples

    def forward_features(self, x, cls=True):
        """ TimeSformer feature propagation
        """
        B = x.shape[0]
        
        x, T, W = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        ## resizing the positional embeddings in case they don't match the input at inference
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0,0,:].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0,1:,:].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)


        ## Time Embeddings
        if self.attention_type != 'space_only':
            cls_tokens = x[:B, 0, :].unsqueeze(1)
            x = x[:,1:]
            x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)
            ## Resizing time embeddings in case they don't match
            if T != self.time_embed.size(1):
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                new_time_embed = new_time_embed.transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.time_embed
            x = self.time_drop(x)
            x = rearrange(x, '(b n) t m -> b (n t) m',b=B,t=T)
            x = torch.cat((cls_tokens, x), dim=1)

        ## Attention blocks
        for blk in self.blocks:
            x = blk(x, B, T, W)
        
        ### Predictions for space-only baseline
        if self.attention_type == 'space_only': # attention over 2nd dim
            x = rearrange(x, '(b t) n m -> b t n m',b=B,t=T)
            x = torch.mean(x, 1) # averaging predictions for every frame

        x = self.norm(x)
        
        if cls:
            return x[:, 0]
        else:
            return x

    def get_pseudo_labels(self, device, text):
        # encode ASR with langauge encoder
        vis_emb = text['clip_vis_feat']
        text_emb = self.text_model.encode_text(text['clip_text_ids'].squeeze(dim=1)) # torch.FloatTensor(x.shape[0], 512).to(x.device) # vis_emb # 
        text_emb = (text_emb + vis_emb) / 2.0
        self.label_emb = self.check_device_norm(self.label_emb, device, norm=True)
        text_emb = text_emb / text_emb.norm(dim=1, keepdim=True)
        teacher_x = text_emb @ self.label_emb.t() / self.temp
        return teacher_x  

    def check_device_norm(self, query_tensor, target_device, norm=False):
        if query_tensor.device != target_device:
            query_tensor = query_tensor.to(target_device)
            if norm:
                query_tensor = query_tensor / query_tensor.norm(dim=1, keepdim=True)
        return query_tensor

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)            
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            if v.shape[-1] != patch_size:
                patch_size = v.shape[-1]
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

@MODEL_REGISTRY.register()
class vit_base_patch16_224_develop(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(vit_base_patch16_224_develop, self).__init__()
        self.pretrained = cfg.MODEL.PRETRAINED
        patch_size = 16
        mlp = cfg.MODEL.MLP
        emb = cfg.TRAIN.LABEL_EMB
        lp = cfg.MODEL.TEXT_LP
        num_seg = cfg.MODEL.NUM_SEG
        extra = cfg.MODEL.EXTRA_TR
        drope = cfg.MODEL.DROP_E
        dpr = cfg.MODEL.DROP_PATH
        depth = cfg.TIMESFORMER.DEPTH
        self.model = VisionTransformer(img_size=cfg.DATA.TRAIN_CROP_SIZE, num_classes=cfg.MODEL.NUM_CLASSES, patch_size=patch_size, \
            embed_dim=768, depth=depth, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), \
            drop_rate=0., attn_drop_rate=0., drop_path_rate=dpr, num_frames=cfg.DATA.NUM_FRAMES, \
            attention_type=cfg.TIMESFORMER.ATTENTION_TYPE, label_emb = emb, mlp = mlp, text_model = cfg.MODEL.TEXT_MODEL, \
            lp=lp,num_seg=num_seg,extra_tr=extra,drope=drope,cfg=cfg,**kwargs)

        self.attention_type = cfg.TIMESFORMER.ATTENTION_TYPE
        self.model.default_cfg = default_cfgs['vit_base_patch16_224']
        self.num_patches = (cfg.DATA.TRAIN_CROP_SIZE // patch_size) * (cfg.DATA.TRAIN_CROP_SIZE // patch_size)
        pretrained_model = cfg.TIMESFORMER.PRETRAINED_MODEL
        if self.pretrained:
            load_pretrained(self.model, num_classes=self.model.num_classes, in_chans=kwargs.get('in_chans', 3), \
                filter_fn=_conv_filter, img_size=cfg.DATA.TRAIN_CROP_SIZE, num_patches=self.num_patches, attention_type=self.attention_type, \
                pretrained_model=pretrained_model, num_frames=cfg.DATA.NUM_FRAMES, pre_num=cfg.MODEL.PRE_CLASSES)
        else:
            print('not loading any pretrained weights!')

    def forward(self, x):
        x = self.model(x)
        return x

    