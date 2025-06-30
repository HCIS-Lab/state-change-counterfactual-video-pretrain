# code from https://github.com/facebookresearch/HierVL/blob/main/run/test_charades.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import tqdm
import pickle
import argparse
import numpy as np
import pandas as pd
import transformers
from csv import reader
from sacred import Experiment
import torch.nn.functional as F
import torch
import sys
# sys.path.append('../')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import model.metric as module_metric
import data_loader.data_loader as module_data
from utils import state_dict_data_parallel_fix
from parse_config import ConfigParser

from transformers import AutoTokenizer, FlavaTextModel

ex = Experiment('test')

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    # a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = F.normalize(a, dim=-1) #a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = F.normalize(b, dim=-1) #b / torch.max(b_n, eps * torch.ones_like(b_n))

    sim_mt = torch.mm(a_norm, b_norm.T)
    return sim_mt

def tokenize(tokenizer, input_str):
    """
    input: pyhon list of strings
    output: dictionary with batched tensors
    """
    # padding due to batched inputs
    tokens = tokenizer(input_str, padding=True, return_tensors="pt")

    return tokens

def embed(embedder, tokens):
    """
    input: dictionary with batched tensors (from tokenizer)
    output: batched tensor
    """

    outputs = embedder(**tokens)
    embedding = outputs.last_hidden_state
    # print(embedding.shape) # [6, L, 768] note that L varies per mini-batch

    return embedding

def charades_map(submission_array, gt_array):
    """
    Approximate version of the charades evaluation function
    For precise numbers, use the submission file with the official matlab script
    """
    fix = submission_array.copy()
    empty = np.sum(gt_array, axis=1) == 0
    fix[empty, :] = np.NINF
    return map(fix, gt_array)

def map(submission_array, gt_array):
    """ Returns mAP, weighted mAP, and AP array """
    m_aps = []
    n_classes = submission_array.shape[1]
    for oc_i in range(n_classes):
        sorted_idxs = np.argsort(-submission_array[:, oc_i])
        tp = gt_array[:, oc_i][sorted_idxs] == 1
        fp = np.invert(tp)
        n_pos = tp.sum()
        if n_pos < 0.1:
            m_aps.append(float('nan'))
            continue
        fp.sum()
        f_pcs = np.cumsum(fp)
        t_pcs = np.cumsum(tp)
        prec = t_pcs / (f_pcs+t_pcs).astype(float)
        avg_prec = 0
        for i in range(submission_array.shape[0]):
            if tp[i]:
                avg_prec += prec[i]
        m_aps.append(avg_prec / n_pos.astype(float))
    m_aps = np.array(m_aps)
    m_ap = np.mean(m_aps)
    w_ap = (m_aps * gt_array.sum(axis=0) / gt_array.sum().sum().astype(float))
    return m_ap, w_ap, m_aps

@ex.main
def run():
    # setup data_loader instances
    config._config['data_loader']['type'] = 'TextVideoDataLoader'
    config._config['data_loader']['args']['split'] = args.split
    config._config['data_loader']['args']['tsfm_split'] = 'test'  # set transform to test split to remove augmentations
    config._config['data_loader']['args']['shuffle'] = False
    config._config['data_loader']['args']['batch_size'] = args.batch_size
    config._config['data_loader']['args']['sliding_window_stride'] = args.sliding_window_stride

    data_loader = config.initialize('data_loader', module_data)

    # tokenizer = transformers.AutoTokenizer.from_pretrained(config['arch']['args']['text_params']['model'])

    # build model architecture
    import model.counterfactual as module_arch
    model = config.initialize('arch', module_arch)

    # get function handles of loss and metrics
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # logger.info('Loading checkpoint: {} ...'.format(config.resume))

    if config.resume is not None:
        print(config.resume)
        checkpoint = torch.load(config.resume)
        state_dict = checkpoint['state_dict']
        new_state_dict = state_dict_data_parallel_fix(state_dict, model.state_dict())
        model.load_state_dict(new_state_dict, strict=False)
    else:
        print('Using random weights')

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    flava_tokenizer = AutoTokenizer.from_pretrained("facebook/flava-full")
    flava_model = FlavaTextModel.from_pretrained("facebook/flava-full").to(device).eval()
    model.eval()

    # construct set of sentences.
    cls_arr = []
    with open('/nfs/charjabug/data/md1/datasets/charades-ego/CharadesEgo/Charades_v1_classes.txt', 'r') as charades:
        csv_reader = list(reader(charades))
    for line in csv_reader:
        cls_arr.append(line[0][5:])
    # print(len(cls_arr))
    token_arr = tokenize(flava_tokenizer, cls_arr)
    # print(len(token_arr))
    text_embeds = embed(flava_model, token_arr.to(device)).cpu().detach()
    # print(text_embeds.shape)

    # output of print():
    # 157
    # 3
    # torch.Size([157, 11, 768])

    # raise Exception("xx")
    meta_arr, gt_arr = [], []
    text_embed_arr = []
    vid_embed_arr = []
    print(len(data_loader))
    with torch.no_grad():
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # data_cls = tokenizer(cls_arr, return_tensors='pt', padding=True, truncation=True)
        # data_cls = {key: val.cuda() for key, val in data_cls.items()}

        # dict_cls = {'text':data_cls, 'video': torch.Tensor(1,4,3,224,224)}
        # text_embed, _ = model(dict_cls, return_embeds=True)
        # text_embeds = text_embed.cpu().detach()
        for i, data in tqdm.tqdm(enumerate(data_loader)):
            meta_arr.append(data['meta'])

            # data['text'] = tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
            # data['text'] = {key: val.cuda() for key, val in data['text'].items()}
            # print()
            # print('-'*20)
            # print(data['target'].shape)
            # print('-'*20)
            # print()
            # print()
            # print('-'*20)
            # print(data['text'])
            # print('-'*20)
            # print()
            if isinstance(data['video'], list):
                data['video'] = [x.to(device) for x in data['video']]
            else:
                data['video'] = data['video'].to(device)

            # try:    # for temporal
            #     _, vid_embed, _ = model(data, return_embeds=True)
            # except:
                # print(data['video'].shape)
                vid_embed, _ = model(data['video'], return_embeds=True)
                # print(vid_embed.shape)

            vid_embed_arr.append(vid_embed.cpu().detach())
            gt_arr.append(data['target'].cpu().detach())
    
    vid_embeds = torch.cat(vid_embed_arr)
    print(vid_embeds.shape)
    gt_embeds = torch.cat(gt_arr)
    print(gt_embeds.shape)
    print(text_embeds[:, 0, :].shape)
    # torch.Size([846, 768])
    # torch.Size([846, 157])
    # torch.Size([157, 11, 768])
    # raise Exception("xx")
    sims = sim_matrix(text_embeds[:, 0, :], vid_embeds)
    print(sims.shape)
    sims = sims.numpy().T
    gt_embeds = gt_embeds.numpy()
    m_ap, w_ap, m_aps = charades_map(np.vstack(sims), np.vstack(gt_embeds))
    print('mAP: {:.3f}'.format(m_ap * 100))

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-r', '--resume',
                      default='/nfs/wattrel/data/md0/datasets/state_aware/results/EgoClip_CF/models/0328_10_23_10/checkpoint-epoch6.pth', # FT checkpoint
                      #default='download/checkpoints/hiervl/hievl_sa.pth', #ZS checkpoint
                      #default='download/checkpoints/hiervl/hievl_sa_2.pth', #ZS best checkpoint
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-gpu', '--gpu', default=0, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-s', '--sliding_window_stride', default=-1, type=int,
                      help='test time temporal augmentation, repeat samples with different start times.')
    args.add_argument('--save_feats', default=None,
                      help='path to store text & video feats, this is for saving embeddings if you want to do offline retrieval.')
    args.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                      help='split to evaluate on.')
    args.add_argument('--batch_size', default=1, type=int,
                      help='size of batch')
    config = ConfigParser(args, test=True, eval_mode='charades')

    # hack to get sliding into config
    args = args.parse_args()
    config._config['sliding_window_stride'] = args.sliding_window_stride
    ex.add_config(config.config)

    os.environ["CUDA_VISIBLE_DEVICES"] =  ""+str(args.gpu)


    ex.run()