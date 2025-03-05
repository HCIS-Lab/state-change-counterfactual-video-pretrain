import os
import torch.nn as nn
from datasets import Breakfast_FRAMES, GTEA_FRAMES, SALADS_FRAMES
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import yaml
from dotmap import DotMap
import pprint
# from as_utils.text_prompt import *
from pathlib import Path
from as_utils.Augmentation import *
from as_utils.load_hiervl import *
from as_utils.load_cf import *
from as_utils.load_milnce import *
from as_utils.load_pvrl import *

import clip
import numpy as np


dataset = 'breakfast'
config = './as_configs/breakfast/breakfast_exfm.yaml'
model_name = 'pvrl'
log_time = ''
class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)


def main():
    # global args, best_prec1
    global best_prec1
    global global_step
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config', '-cfg', default='./as_configs/breakfast/breakfast_exfm.yaml')
    # parser.add_argument('--log_time', default='')
    # parser.add_argument('--dataset', default='breakfast')
    # parser.add_argument('--model', default='hiervl')
    # args = parser.parse_args()


    with open(config, 'r') as f:
        config = yaml.safe_load(f)
    working_dir = os.path.join('./exp', config['network']['type'], config['network']['arch'], config['data']['dataset'],
                               log_time)
    print('-' * 80)
    print(' ' * 20, "working dir: {}".format(working_dir))
    print('-' * 80)

    print('-' * 80)
    print(' ' * 30, "Config")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    print('-' * 80)

    config = DotMap(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.

    if model_name == 'clip':
        model, clip_state_dict = clip.load(config.network.arch, device=device, jit=False, tsm=config.network.tsm,
                                       T=config.data.num_segments, dropout=config.network.drop_out,
                                       emb_dropout=config.network.emb_dropout, if_proj=config.network.if_proj)
    # Must set jit=False for training  ViT-B/32
        model = ImageCLIP(model)
        model = torch.nn.DataParallel(model).cuda()
        clip.model.convert_weights(model)
        if config.pretrain:
            if os.path.isfile(config.pretrain):
                print(("=> loading checkpoint '{}'".format(config.pretrain)))
                checkpoint = torch.load(config.pretrain)
                model.load_state_dict(checkpoint['model_state_dict'])
                del checkpoint
            else:
                print(("=> no checkpoint found at '{}'".format(config.pretrain)))

    elif model_name == 'hiervl':
        model = load_hiervl("/nfs/wattrel/data/md0/datasets/state_aware/pretrained/hievl_sa.pth")
        model = torch.nn.DataParallel(model).cuda()
    elif model_name == 'cf':
        # /nfs/wattrel/data/md0/datasets/state_aware/results/a_2e5_2nd_2025-02-21_13_53_56/results/EgoClip_4f/models/4391831
        # /N/project/ego4d_vlm/state-aware-video-pretrain/experiments/a_2e5_2025-02-16_20_51_14/results/EgoClip_4f/models/4362317/checkpoint-epoch3.pth
        model = load_cf("/nfs/wattrel/data/md0/datasets/state_aware/results/a_2e5_2nd_2025-02-21_13_53_56/results/EgoClip_4f/models/4391831/checkpoint-epoch6.pth")
        # model = load_cf("/nfs/wattrel/data/md0/datasets/state_aware/results/EgoClip_CF/models/0225_12_21_14/cf_7e6_16b_epoch5.pth")
        #model = load_cf("/nfs/wattrel/data/md0/datasets/state_aware/results/EgoClip_CF/models/0215_22:03:20/checkpoint-epoch9.pth")
        # model = load_cf("/nfs/wattrel/data/md0/datasets/state_aware/results/EgoClip_CF/models/0215_22:03:20/checkpoint-epoch9.pth")
        model = torch.nn.DataParallel(model).cuda()
    elif model_name == 'pvrl':
        model = load_pvrl("/nfs/wattrel/data/md0/datasets/state_aware/pvrl/model_epoch_00025.pyth")
    elif model_name == 'milnce':
        model = load_milnce()
        model = torch.nn.DataParallel(model).cuda()

    transform_val = get_augmentation(False, config)

    if dataset == 'breakfast':
        val_data = Breakfast_FRAMES(transforms=transform_val)
    elif dataset == 'gtea':
        val_data = GTEA_FRAMES(transform=transform_val)
    elif dataset == 'salads':
        val_data = SALADS_FRAMES(transform=transform_val)
    else:
        val_data = None
    val_loader = DataLoader(val_data, batch_size=config.data.batch_size, num_workers=20,
                            shuffle=False, pin_memory=False, drop_last=False)


    model.eval()
    save_dir = config.data.save_dir
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    if dataset == 'gtea':
        non_splt = False
    else:
        non_splt = True

    if model_name == 'clip':
        with torch.no_grad():
            for iii, (image, filename) in enumerate(tqdm(val_loader)):
                if not os.path.exists(os.path.join(save_dir, filename[0])):
                    if non_splt:
                        image = image.view((-1, config.data.num_frames, 3) + image.size()[-2:])
                    else:
                        image = image.view((1, -1,  3) + image.size()[-2:])
                    b, t, c, h, w = image.size()
                    image_input = image.view(b * t, c, h, w)
                    if non_splt:
                        image_inputs = image_input.to(device)
                        image_features = model(image_inputs)
                        image_features = image_features.view(b, t, -1)
                        for bb in range(b):
                            np.save(os.path.join(save_dir, filename[bb]), image_features[bb, :].cpu().numpy())
                    else:
                        image_inputs = torch.split(image_input, 1024)
                        image_features = []
                        for inp in image_inputs:
                            inp = inp.to(device)
                            image_features.append(model.encode_image(inp))
                        image_features = torch.cat(image_features)
                        np.save(os.path.join(save_dir, filename[0]), image_features.cpu().numpy())
    else:
        with torch.no_grad():
            for iii, (window, filename) in enumerate(tqdm(val_loader)):
                if not os.path.exists(os.path.join(save_dir, filename[0])):
                    
                    if non_splt :
                        window_size = 15
                        if model_name == 'milnce':
                            window = 16
                        # [b, 96, 224, 224]
                        window = window.view((-1, config.data.num_frames, 3) + window.size()[-2:])
                        # [b, 32, 3, 224, 224]
                        b, T_padded, c, h, w = window.shape
                        # window = window.to(device)
                        # [1, 15, 3, 224, 224]
                        first_frame = window[:, 0:1, :, :, :]  # Shape: [b, 1, C, H, W]
                        last_frame = window[:, -1:, :, :, :]   # Shape: [b, 1, C, H, W]

                        if window_size % 2 == 1:
                            front_padding = first_frame.repeat(1, (window_size-1)//2, 1, 1, 1)

                            rear_padding = last_frame.repeat(1, (window_size-1)//2, 1, 1, 1)
                        else:
                            front_pad = window_size // 2 - 1
                            rear_pad  = window_size // 2
                            front_padding = first_frame.repeat(1, front_pad, 1, 1, 1)
                            rear_padding  = last_frame.repeat(1, rear_pad, 1, 1, 1)

                        padded_video = torch.cat([front_padding, window, rear_padding], dim=1)
                        # [b, T+14, c, h , w]
                        b, T, C, H, W = padded_video.shape  # T = 24 (10 original + 7 front + 7 rear)
                        padded_video = padded_video.unfold(dimension=1, size=window_size, step=1)  # Shape: [b, T_new, window_size, C, H, W]
                        window = padded_video.reshape(-1, window_size, 3, h, w)
                        
                        window = window.to(device)
                        if model_name == 'hiervl' or 'cf' in model:
                            feature = model(data=window, video_only=True)
                        elif model_name == 'milnce':
                            window = window.permute(0,2,1,3,4)
                            feature = model(window)
                            feature = feature['mixed_5c']

                        # [b*num_frames, c]
                        feature = feature.reshape(b, config.data.num_frames, -1)
                        feature = feature.permute(0,2,1)
                        
                        for bb in range(b):
                            np.save(os.path.join(save_dir, filename[bb]), feature[bb, :].cpu().numpy())
                        # image_features = model(window)
                        # image_features = image_features.view(b, t, -1)
                        # for bb in range(b):
                        #     np.save(os.path.join(save_dir, filename[bb]), image_features[bb, :].cpu().numpy())
                    else:
                        window_size = 15
                        # [b, c*windows, t, h, w]
                        b, win_c , h, w = window.size()
                        window = window.reshape(1, -1, 3, h, w)
                        b, T_padded, c, h, w = window.shape
                        window = window.as_strided(
                            size=(b, T_padded - window_size + 1, window_size, c, h, w),  # [b, T, 21, c, h, w]
                            stride=(window.stride(0), window.stride(1), window.stride(1), window.stride(2), window.stride(3), window.stride(4))
                        )
                        window = window.reshape(-1, window_size, 3, h, w)
                        sub_batches = torch.split(window, 8, dim=0)

                        feature_list = []
                        for i, sb in enumerate(sub_batches):
                            sb = sb.to(device)
                            sb_features = model(video=sb, video_only=True)
                            feature_list.append(sb_features)

                        feature = torch.stack(feature_list[:-1], dim=0)
                        b, _, c = feature.shape
                        feature = feature.reshape(-1, c)
                        feature = torch.cat((feature, feature_list[-1]), dim=0)
                        feature = feature.permute(1,0)
                        np.save(os.path.join(save_dir, filename[0]), feature.cpu().numpy())

if __name__ == '__main__':
    main()
