from transformers import ViTModel, ViTConfig
import torch.nn as nn
import torch
from collections import OrderedDict
import torch.nn.functional as F

class outp(nn.Module):
    def __init__(self, clip_length=64, embed_dim=256, n_layers=6):
        super(outp, self).__init__()
        self.clip_length = clip_length
        drop_rate = 0.
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4)
        self.transformer_enc = nn.TransformerEncoder(enc_layer, num_layers=4, norm=nn.LayerNorm(
            embed_dim))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, clip_length + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

    def forward(self, x):
        nvids = x.shape[0]

        cls_tokens = self.cls_token.expand(nvids, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x.transpose_(1, 0)
        o = self.transformer_enc(x)

        return o[0]

class inp(nn.Module):
    def __init__(self, clip_length=None, embed_dim=2048, n_layers=6):
        super(inp, self).__init__()

        self.attn = nn.MultiheadAttention(256, 4)
        self.ln_1 = nn.LayerNorm(256)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(256, 256 * 4)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(256 * 4, 256))
        ]))
        self.ln_2 = nn.LayerNorm(256)
        self.attn = nn.MultiheadAttention(256, 4)
    def attention(self, x: torch.Tensor):
        return self.attn(x, x, x, need_weights=False)[0]
    def forward(self, x: torch.Tensor):
        b, t, c = x.size()
        x = x.contiguous()


        x_original = x

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x + x_original

        return x.mean(dim=1, keepdim=False)
    
hidden_dim=768

class vit_classifier(nn.Module):
    def __init__(self, num_frames, num_segments, device):
        super().__init__()

        # # Define the first ViT for frame fusion (merges 8 frames -> 256 features)
        # self.vit_frames = ViTModel(ViTConfig(
        #     image_size=num_frames,  # Treats the 8 frames as "patches"
        #     num_channels=256,  # Each frame has 256 features
        #     hidden_size=256,  # Keep hidden dim the same as input dimx
        #     num_hidden_layers=4,  # Use a reasonable depth
        #     num_attention_heads=4
        # ))
        # decoder_in = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward)
        # decoder_out = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward)
        # self.tin = nn.TransformerDecoder(decoder_in, num_layers=num_decoder_layers)
        # self.tout = nn.TransformerDecoder(decoder_out, num_layers=num_decoder_layers)

        # self.norm= nn.LayerNorm(hidden_dim)
        # self.norm_1 = nn.LayerNorm(hidden_dim)
        # self.norm_2 = nn.BatchNorm1d(256)
        self.tin = nn.Transformer(d_model=hidden_dim, nhead=2, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=256, dropout=0.1, batch_first=True)
        self.tout = nn.Transformer(d_model=hidden_dim, nhead=2, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=256, dropout=0.1, batch_first=True)
        # self.in_proj = nn.Linear(256, 256)

        # # Define the second ViT for clip fusion (merges 64 clips -> final output)
        # self.vit_clips = ViTModel(ViTConfig(
        #     image_size=num_segments,  # Treats the 64 clips as "patches"
        #     num_channels=256,  # Each clip has 256 features
        #     hidden_size=256,  # Keep hidden dim the same as input dimx
        #     num_hidden_layers=4,  # Use a reasonable depth
        #     num_attention_heads=4
        # ))
        # self.inn = inp()
        # self.out = outp()

        self.proj = nn.Linear(hidden_dim, 10)  # Final classification layer
        self.device = device

    def forward(self, x):
        # print("x0 ", x.shape)
        b, n, f, d = x.size()  # (16, 64, 8, 256)
        # x = F.normalize(x, dim=-1)

        # Merge 8 frames into a single feature vector per clip
        # x = x.view(-1, d)
        # x = self.norm(x)
        x = x.view(-1, f, d)  # (256, 8, 256)
        # x = self.in_proj(x)
        # d = 256
        # print("x1 ", x.shape)

        # x = self.vit_frames(pixel_values=x).last_hidden_state[:, 0, :]  # Extract CLS token
        # x = self.inn(x)
        x = self.tin(x, x)
        x = x.mean(dim=1, keepdim=False)
        # x = self.norm_1(x)
        # print("x2 ", x.shape)  # (256, 256)

        # Merge 64 clips into a single feature vector per video
        x = x.view(b, n, d)  # (16, 64, 256)
        # x = self.norm_1(x)
        # print("x3 ", x.shape)
        x = self.tout(x, x)
        # x = self.vit_clips(pixel_values=x).last_hidden_state[:, 0, :]  # Extract CLS token
        # print("x4 ", x.shape)  # (16, 256)
        x = x.mean(dim=1, keepdim=False)
        # x = self.norm_2(x)

        return self.proj(x)  # (16, 10)
