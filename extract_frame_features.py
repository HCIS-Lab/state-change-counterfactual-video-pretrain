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

import numpy as np


class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)


def main():
    global args, best_prec1
    global global_step
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='./as_configs/gtea/gtea_exfm.yaml')
    parser.add_argument('--log_time', default='')
    parser.add_argument('--dataset', default='gtea')
    parser.add_argument('--model', default='cf')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    working_dir = os.path.join('./exp', config['network']['type'], config['network']['arch'], config['data']['dataset'],
                               args.log_time)
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

    if args.model == 'CLIP':
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

    elif args.model == 'hiervl':
        model = load_hiervl("/nfs/wattrel/data/md0/datasets/state_aware/pretrained/hievl_sa.pth")
        model = torch.nn.DataParallel(model).cuda()
    elif args.model == 'cf':
        model = load_cf("/nfs/wattrel/data/md0/datasets/state_aware/results/EgoClip_CF/models/0215_22:03:20/checkpoint-epoch9.pth")
        model = torch.nn.DataParallel(model).cuda()
    # elif args.model == 'pvr':
    # elif args.model == '':

    transform_val = get_augmentation(False, config)

    if args.dataset == 'breakfast':
        val_data = Breakfast_FRAMES(transforms=transform_val)
    elif args.dataset == 'gtea':
        val_data = GTEA_FRAMES(transform=transform_val)
    elif args.dataset == 'salads':
        val_data = SALADS_FRAMES(transform=transform_val)
    else:
        val_data = None
    val_loader = DataLoader(val_data, batch_size=config.data.batch_size, num_workers=20,
                            shuffle=False, pin_memory=False, drop_last=False)


    

    model.eval()
    save_dir = config.data.save_dir
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    if args.dataset == 'gtea':
        non_splt = False
    else:
        non_splt = True

    if args.model == 'clip':
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
                    sub_batches = torch.split(window, 2, dim=0)
                    # if non_splt:
                    #     window = window.view((-1, config.data.num_frames, 3) + window.size()[-2:])
                    # else:
                    #     window = window.view((1, -1,  3) + window.size()[-2:])
                    
                    # image_input = window.view(b * t, c, h, w)

                    if non_splt :
                        feature_list = []
                        for i, sb in enumerate(sub_batches):
                            sb = sb.to(device)
                            sb_features = model(sb)
                            #print(sb_features.shape)
                            feature_list.append(sb_features)

                        feature = torch.stack(feature_list[:-1], dim=0)
                        b, _, c = feature.shape
                        feature = feature.reshape(-1, c)
                        feature = torch.cat((feature, feature_list[-1]), dim=0)
                        feature = feature.permute(1,0)
                        np.save(os.path.join(save_dir, filename[0]), feature.cpu().numpy())
                    else:
                        image_inputs = image_input.to(device)
                        image_features = model_image(image_inputs)
                        image_features = image_features.view(b, t, -1)
                        for bb in range(b):
                            np.save(os.path.join(save_dir, filename[bb]), image_features[bb, :].cpu().numpy())

if __name__ == '__main__':
    main()
