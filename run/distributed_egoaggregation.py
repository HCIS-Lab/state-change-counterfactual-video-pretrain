# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models

import transformers
from tensorboardX import SummaryWriter
import data_loader.data_loader as module_data
from trainer import Multi_Trainer_dist_EgoAgg
import model.loss as module_loss
import model.metric as module_metric
import model.counterfactual as module_arch
import utils.visualizer as module_vis
from sacred import Experiment
ex = Experiment('train')
import collections
from parse_config import ConfigParser
import utils.visualizer as module_vis
from utils.util import replace_nested_dict_item, load_checkpoint_after_preemption

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

best_acc1 = 0

def init_dataloaders(config, module_data, data_loader_type="data_loader"): #data_loader_type can be one of ["data_loader", "aggregate_data_loader"]
    """
    We need a way to change split from 'train' to 'val'.
    """
    if "type" in config[data_loader_type] and "args" in config[data_loader_type]:
        # then its a single dataloader
        data_loader = [config.initialize(data_loader_type, module_data)]
        config[data_loader_type]['args'] = replace_nested_dict_item(config[data_loader_type]['args'], 'split', 'val')
        config[data_loader_type]['args'] = replace_nested_dict_item(config[data_loader_type]['args'], 'batch_size', 1)
        valid_data_loader = [config.initialize(data_loader_type, module_data)]
    elif isinstance(config[data_loader_type], list):
        data_loader = [config.initialize(data_loader_type, module_data, index=idx) for idx in
                       range(len(config[data_loader_type]))]
        new_cfg_li = []
        for dl_cfg in config[data_loader_type]:
            dl_cfg['args'] = replace_nested_dict_item(dl_cfg['args'], 'split', 'val')
            dl_cfg['args'] = replace_nested_dict_item(dl_cfg['args'], 'batch_size', 1)
            new_cfg_li.append(dl_cfg)
        config._config[data_loader_type] = new_cfg_li
        valid_data_loader = [config.initialize(data_loader_type, module_data, index=idx) for idx in
                             range(len(config[data_loader_type]))]
    else:
        raise ValueError("Check data_loader config, not correct format.")

    return data_loader, valid_data_loader
    # return data_loader


def find_free_port():
    import socket
    s = socket.socket()
    s.bind(('', 0))            # Bind to a free port provided by the host.
    return s.getsockname()[1]  # Return the port number assigned.


def main():

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    #parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
    #                    metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-file', default=None, type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')
    ###################################################################
    parser.add_argument('-c', '--config', default='configs/pt/egoclip.json', type=str,
                      help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    parser.add_argument('-o', '--observe', action='store_true',
                      help='Whether to observe (neptune)')
    parser.add_argument('-l', '--launcher', choices=['none', 'pytorch'], default='none',help='job launcher')
    parser.add_argument('-lr1', '--learning_rate1', type=float, default=2e-4)
    parser.add_argument('-sc', '--schedule', default=[60, 80])

    #######################
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size')),
    ]
    config = ConfigParser(parser, options)
    recovered_checkpoint, recovered_epoch = load_checkpoint_after_preemption(config)
    if recovered_checkpoint is not None:
        config["arch"]["args"]["load_checkpoint"] = recovered_checkpoint
        config["trainer"]["start_epoch"] = recovered_epoch + 1
    args = parser.parse_args()
    ex.add_config(config._config)
    ##########################

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    # slurm available
    import os
    if args.world_size == -1 and "SLURM_NPROCS" in os.environ:
        args.world_size = int(os.environ["SLURM_NPROCS"])
        args.rank = int(os.environ["SLURM_PROCID"])
        jobid = os.environ["SLURM_JOBID"]
        try:
            restart_count = os.environ["SLURM_RESTART_COUNT"]
        except:
            restart_count = '0'
        hostfile = "dist_url." + jobid  + '.' + restart_count + ".txt"
        if args.dist_file is not None:
            args.dist_url = "file://{}.{}".format(os.path.realpath(args.dist_file), jobid)
        elif args.rank == 0:
            import socket
            ip = socket.gethostbyname(socket.gethostname())
            port = find_free_port()
            args.dist_url = "tcp://{}:{}".format(ip, port)
            with open(hostfile, "w") as f:
                f.write(args.dist_url)
        else:
            import os
            import time
            while not os.path.exists(hostfile):
                time.sleep(1)
            with open(hostfile, "r") as f:
                args.dist_url = f.read()
        print("dist-url:{} at PROCID {} / {}".format(args.dist_url, args.rank, args.world_size))

    # if args.dist_url == "env://" and args.world_size == -1:
    #     args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, config))
    else:
        raise NotImplementedError
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, config)


def main_worker(gpu, ngpus_per_node, args, config): #TODO: Take config as input
    global best_acc1
    args.gpu = gpu

    if config['visualizer']['type'] != "":
        visualizer = config.initialize(
            name='visualizer',
            module=module_vis,
            exp_name=config['name'],
            web_dir=config._web_log_dir
        )
    else:
        visualizer = None
    logger = config.get_logger('train')

    args.learning_rate1 = config['optimizer']['args']['lr']

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            os.environ["LOCAL_RANK"] = str(args.gpu)
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # build tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config['arch']['args']['text_params']['model'],
                                                               TOKENIZERS_PARALLELISM=False)

    print('Current directory is : {}'.format(os.path.abspath(os.getcwd())))
    # setup data_loader instances
    print('ARGS.RANK NOW IS {}'.format(args.rank))
    print('CONFIG NOW IS {}'.format(config))
    config.args = args
    data_loader, valid_data_loader = init_dataloaders(config, module_data)
    agg_data_loader, agg_valid_data_loader = init_dataloaders(config, module_data, data_loader_type="aggregate_data_loader")
    # data_loader = init_dataloaders(config, module_data)
    # agg_data_loader = init_dataloaders(config, module_data, data_loader_type="aggregate_data_loader")
    if args.rank == 0:
        print('Train dataset: ', [x.n_samples for x in data_loader], ' samples')
        # print('Val dataset: ', [x.n_samples for x in valid_data_loader], ' samples')
        print('Agg Train dataset: ', [x.n_samples for x in agg_data_loader], ' samples')
        # print('Agg Val dataset: ', [x.n_samples for x in agg_valid_data_loader], ' samples')
    # build model architecture, then print to console

    model = config.initialize('arch', module_arch)

    ###########################################
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        raise ValueError
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    ###############################################

    if args.rank == 0:
        logger.info(model)

    # get function handles of loss and metrics
    loss = config.initialize(name="loss", module=module_loss)
    # Add additional losses based on hierarchical requirements
    intra_modal_video_loss = config.initialize(name="hierarchical_loss", module=module_loss) if config["training_methods"]["hierarchical"]["intra-modal"] else None
    intra_modal_text_loss = config.initialize(name="hierarchical_loss", module=module_loss) if config["training_methods"]["hierarchical"]["intra-modal"] else None
    inter_parent_video_loss = config.initialize(name="hierarchical_loss", module=module_loss) if config["training_methods"]["hierarchical"]["inter-modal"] else None
    inter_parent_text_loss = config.initialize(name="hierarchical_loss", module=module_loss) if config["training_methods"]["hierarchical"]["inter-modal"] else None
    additional_losses = [intra_modal_video_loss, intra_modal_text_loss, inter_parent_video_loss, inter_parent_text_loss]

    metrics = [getattr(module_metric, met) for met in config['metrics']]
    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.initialize('optimizer', transformers, trainable_params)
    lr_scheduler = None
    if 'lr_scheduler' in config._config:
        if hasattr(transformers, config._config['lr_scheduler']['type']):
            lr_scheduler = config.initialize('lr_scheduler', transformers, optimizer)
        else:
            print('lr scheduler not found')
    if config['trainer']['neptune']:
        writer = ex
    else:
        writer = None

    if args.rank == 0:
        writer = SummaryWriter(log_dir=str(config.tf_dir))

    trainer = Multi_Trainer_dist_EgoAgg(args, model, loss, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      agg_data_loader=agg_data_loader,
                      agg_valid_data_loader=agg_valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      visualizer=visualizer,
                      writer=writer,
                      tokenizer=tokenizer,
                      max_samples_per_epoch=config['trainer']['max_samples_per_epoch'],
                      additional_losses=additional_losses,
                      start_epoch=config['trainer']['start_epoch'])

    trainer.train()

if __name__ == '__main__':
    main()