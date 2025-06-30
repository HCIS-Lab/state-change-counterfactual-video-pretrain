import torch
 
from model import *
from batch_gen import BatchGenerator
from eval import func_eval

import os
import argparse
import numpy as np
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 19980125 # my birthday, :)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
 
parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--feature', default='cf') #cf_1e5_18b_10epoch cf_7e6_16b_epoch5
parser.add_argument('--dataset', default="50salads")
parser.add_argument('--split', default='1')
parser.add_argument('--model_dir', default='models')
parser.add_argument('--result_dir', default='results')
parser.add_argument('--path', default=None)

args = parser.parse_args()


num_epochs = 120

lr = 0.0005
num_layers = 10
num_f_maps = 64
if args.feature == 'i3d':
    features_dim = 2048
elif 'cf' in args.feature:
    features_dim = 768
elif args.feature == 'hiervl':
    features_dim = 256
elif args.feature == 'milnce':
    features_dim = 1024
elif args.feature == 'clip':
    features_dim = 768

bz = 1

channel_mask_rate = 0.3


# use the full temporal resolution @ 15fps
sample_rate = 1
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
if args.dataset == "50salads":
    sample_rate = 2

# To prevent over-fitting for GTEA. Early stopping & large dropout rate
if args.dataset == "gtea":
    channel_mask_rate = 0.5
    
if args.dataset == 'breakfast':
    lr = 0.0001
    num_epochs = 20


vid_list_file = "/nfs/wattrel/data/md0/kung/state-aware-video-pretrain/data/"+args.dataset+"/splits/train.split"+args.split+".bundle"
vid_list_file_tst = "/nfs/wattrel/data/md0/kung/state-aware-video-pretrain/data/"+args.dataset+"/splits/test.split"+args.split+".bundle"
features_path = "/nfs/wattrel/data/md0/datasets/action_seg_datasets/"+args.dataset+"/" +args.feature+'_split'+args.split+'/'
if args.dataset == 'breakfast':
    features_path = features_path + 'combined_feat/'
gt_path = "/nfs/wattrel/data/md0/datasets/action_seg_datasets/data/"+args.dataset+"/groundTruth/"
 
if args.path !=None:
    args.path =  os.path.join('models', args.feature, args.dataset, 'split_'+args.split, 'epoch-'+str(args.path)+'.model')

mapping_file = "/nfs/wattrel/data/md0/datasets/action_seg_datasets/data/"+args.dataset+"/mapping.txt"
 
model_dir = os.path.join("models", args.feature, args.dataset, "split_"+args.split)
os.makedirs(model_dir, exist_ok=True)
results_dir = os.path.join("results", args.feature, args.dataset, "split_"+args.split)
os.makedirs(results_dir, exist_ok=True)
 
 
file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])
index2label = dict()
for k,v in actions_dict.items():
    index2label[v] = k
num_classes = len(actions_dict)


trainer = Trainer(num_layers, 2, 2, num_f_maps, features_dim, num_classes, channel_mask_rate)
if args.action == "train":
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen.read_data(vid_list_file)

    batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen_tst.read_data(vid_list_file_tst)

    trainer.train(model_dir, batch_gen, num_epochs, bz, lr, batch_gen_tst, args.path)

if args.action == "predict":
    batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen_tst.read_data(vid_list_file_tst)
    trainer.predict(model_dir, results_dir, features_path, batch_gen_tst, num_epochs, actions_dict, sample_rate)

