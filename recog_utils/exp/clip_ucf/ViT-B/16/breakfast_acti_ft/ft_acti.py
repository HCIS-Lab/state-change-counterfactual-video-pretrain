import os
import torch.nn as nn
from datasets.datasets import Breakfast_feat
from torch.utils.data import DataLoader
from tqdm import tqdm
# import wandb
import argparse
import shutil
from pathlib import Path
import yaml
from dotmap import DotMap
import pprint
# from modules.fusion_module import vit_classifier
# from modules.fusion_module import fusion_base
from modules.vit_classifier import vit_classifier
import torch.optim as optim
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from utils.Augmentation import *
from utils.solver import _lr_scheduler
from utils.tools import *
from utils.text_prompt import *
from utils.saving import *
import torch
import random
import numpy as np

SEED = 1024
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

@torch.no_grad()
def eval_model(model, data_loader, device):
    # Evaluate the model on data from valloader
    correct = 0
    total = 0

    # fusion_model, fusion_model_up, proj = model
    # fusion_model.eval()
    # fusion_model_up.eval()
    # proj.eval()
    model.eval()

    assert not torch.is_grad_enabled(), "grad is enabled during inference"

    for images, labels in data_loader:
        # b, n, f, d = images.size()
        images, labels = images.to(device, dtype=torch.float32), labels.to(device, dtype=torch.long)
        # image_embedding = images.view(-1, f, 256)
        # image_embedding = fusion_model(image_embedding)
        # image_embedding = image_embedding.view(b, n, 256)
        # image_embedding = fusion_model_up(image_embedding)
        # preds = proj(image_embedding)
        preds = model(images)

        total += labels.size(0)
        _, predicted = torch.max(preds.data, 1)
        correct += (predicted == labels).sum().item()

    assert not torch.is_grad_enabled(), "grad is enabled during inference"
    return 100 * correct / total

def main():
    global args, best_prec1
    global global_step
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='./configs/breakfast/breakfast_acti_ft.yaml')
    parser.add_argument('--log_time', default='')
    parser.add_argument('--name', default='Transcls_ls')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    working_dir = os.path.join('./exp', config['network']['type'], config['network']['arch'], config['data']['dataset'],
                               args.log_time)
    # wandb.init(project=config['network']['type'],
    #            name='{}_{}_{}_{}_{}'.format(args.log_time, config['network']['type'], config['network']['arch'],
    #                                      config['data']['dataset'], args.name))
    print('-' * 80)
    print(' ' * 20, "working dir: {}".format(working_dir))
    print('-' * 80)

    print('-' * 80)
    print(' ' * 30, "Config")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    print('-' * 80)

    config = DotMap(config)

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, working_dir)
    shutil.copy('ft_acti.py', working_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.

    # model, clip_state_dict = clip.load(config.network.arch, device=device, jit=False, tsm=config.network.tsm,
    #                                    T=config.data.num_segments, dropout=config.network.drop_out,
    #                                    emb_dropout=config.network.emb_dropout, pretrain=config.network.init,
    #                                    joint=config.network.joint)  # Must set jit=False for training  ViT-B/32
    
    vit_cls = vit_classifier(config.data.num_frames, config.data.num_segments, device)
    vit_cls = vit_cls.to(device)
    # fusion_model = fusion_base('Transf', clip_state_dict, config.data.num_frames).to(device)
    # fusion_model_up = fusion_base("Transf_cls", clip_state_dict, config.data.num_segments).to(device)
    # proj = nn.Linear(256, 10).to(device)

    # for param in vit_cls.parameters():
    #     param.requires_grad = True
    #     param.data = param.data.float()
    #     if param.grad is not None:
    #         param.grad.data = param.grad.data.float()
    
    # for param in vit_cls.fusion_model.parameters():
    #     param.requires_grad = True
    # for param in vit_cls.fusion_model_up.parameters():
    #     param.requires_grad = True

    
    # del model, clip_state_dict

    train_data = Breakfast_feat(mode='train', num_frames=config.data.num_frames,
                                n_split=config.data.n_split, n_seg=config.data.num_segments)
    train_loader = DataLoader(train_data, batch_size=config.data.batch_size, num_workers=config.data.workers,
                              shuffle=True, pin_memory=True, drop_last=True)
    val_data = Breakfast_feat(mode='val', num_frames=config.data.num_frames,
                              n_split=config.data.n_split, n_seg=config.data.num_segments)
    val_loader = DataLoader(val_data, batch_size=config.data.batch_size, num_workers=config.data.workers, shuffle=False,
                            pin_memory=True, drop_last=False)
    # print(train_data.classes)
    # num_params = sum(p.numel() for p in fusion_model.parameters() if p.requires_grad)
    # print(f"Total trainable parameters: {num_params}")
    # num_params = sum(p.numel() for p in fusion_model_up.parameters() if p.requires_grad)
    # print(f"Total trainable parameters: {num_params}")
    # num_params = sum(p.numel() for p in proj.parameters() if p.requires_grad)
    # print(f"Total trainable parameters: {num_params}")
    # print('#'*20)
    
    num_params = sum(p.numel() for p in vit_cls.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params}")
    # num_params = sum(p.numel() for p in vit_cls.fusion_model.parameters() if p.requires_grad)
    # print(f"Total trainable parameters: {num_params}")
    # num_params = sum(p.numel() for p in vit_cls.fusion_model_up.parameters() if p.requires_grad)
    # print(f"Total trainable parameters: {num_params}")
    # num_params = sum(p.numel() for p in vit_cls.proj.parameters() if p.requires_grad)
    # print(f"Total trainable parameters: {num_params}")

    # raise Exception("")


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(vit_cls.parameters(), 
        betas=(0.9, 0.98), lr=config.solver.lr, eps=1e-8, weight_decay=config.solver.weight_decay)
    # optimizer = optim.AdamW([
    #     {"params": fusion_model.parameters()},
    #     {"params": fusion_model_up.parameters()},
    #     {"params": proj.parameters()}], 
    #     betas=(0.9, 0.98), lr=config.solver.lr, eps=1e-8, weight_decay=config.solver.weight_decay)
    lr_scheduler = _lr_scheduler(config, optimizer)
    
    all_test = []
    for epoch in range(0, config.solver.epochs-1):
        running_loss = 0.0
        correct = 0
        # fusion_model.train()
        # fusion_model_up.train()
        # proj.train()
        vit_cls.train()

        for _, (image_embedding, labels) in enumerate(tqdm(train_loader)):
            image_embedding = image_embedding.to(device, dtype=torch.float32)
            # b, n, f, d = image_embedding.size()
            labels = labels.to(device, dtype=torch.long)
            
            with torch.set_grad_enabled(True):
                # image_embedding = image_embedding.view(-1, f, 256)
                # image_embedding = fusion_model(image_embedding)
                # image_embedding = image_embedding.view(b, n, 256)
                # image_embedding = fusion_model_up(image_embedding)
                # preds = proj(image_embedding)
                preds = vit_cls(image_embedding)
            
            loss = criterion(preds, labels)
            
            # SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            with torch.no_grad():
                _, predicted = torch.max(preds.detach().data, 1)
                correct += (predicted == labels).sum().item()

        lr_scheduler.step()
        print('epoch - %d loss: %.3f accuracy: %.3f' % (epoch, running_loss / len(train_loader), 100 * correct / len(train_loader.dataset)))
        
        with torch.no_grad():
            # test_acc = eval_model([fusion_model, fusion_model_up, proj], val_loader, device)
            test_acc = eval_model(vit_cls, val_loader, device)
            all_test.append(test_acc)
            print("\ttest acc: %.2f" % (test_acc))

        # epoch_saving(epoch, model, fusion_model, optimizer, filename)
        # if is_best:
        #     best_saving(working_dir, epoch, model, fusion_model, optimizer)
    print()
    print("Done training")
    print("best epoch: ", max(all_test))
    print("\tat: ", all_test.index(max(all_test)))


if __name__ == '__main__':
    main()