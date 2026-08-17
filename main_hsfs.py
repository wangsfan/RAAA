# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

# Modified for DyTox by Arthur Douillard
import argparse
import copy
import datetime
import json
import os
import random
import statistics
import time
import warnings
from pathlib import Path
import yaml
from continual.pod import _local_pod

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from continual.mixup import Mixup
import continual.utils_iirc as utils
from continual import factory_iirc_hcast
from continual import scaler
from continual.classifier import Classifier
from continual.sam import SAM
from continual.engine_iirc_hcast import train_one_epoch, eval_and_log_in, eval_and_log_end
from continual.losses_iirc import bce_with_logits, soft_bce_with_logits
from iirc.datasets_loader_hcast import get_lifelong_datasets
from lifelong_methods.buffer.buffer_hcast import Buffer, TaskDataMergedWithBuffer, get_finetuning_dataset

warnings.filterwarnings("ignore")


def get_transforms(dataset_name):
    essential_transforms_fn = None
    augmentation_transforms_fn = None
    if "cifar100" in dataset_name:
        essential_transforms_fn = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
        ])
        augmentation_transforms_fn = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
        ])
    elif "imagenet" in dataset_name:
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        essential_transforms_fn = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            normalize,
        ])
        augmentation_transforms_fn = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            normalize,
        ])
    return essential_transforms_fn, augmentation_transforms_fn

def get_args_parser():
    parser = argparse.ArgumentParser('DyTox training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--incremental-batch-size', default=None, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--base-epochs', default=500, type=int,
                        help='Number of epochs for base task')
    parser.add_argument('--no-amp', default=False, action='store_true',
                        help='Disable mixed precision')

    # Model parameters
    parser.add_argument('--model', default='')
    parser.add_argument('--input-size', default=32, type=int, help='images input size')
    parser.add_argument('--patch-size', default=16, type=int)
    parser.add_argument('--embed-dim', default=768, type=int)
    # parser.add_argument('--depth', default=12, type=int)
    parser.add_argument('--num-heads', default=12, type=int)
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--norm', default='layer', choices=['layer', 'scale'],
                        help='Normalization layer type')
    # HCAST parameters
    parser.add_argument('--n-segments', default=64, type=int,
                        help='Number of segments for HCAST, with 64 for CIFAR and 196 for ImageNet')
    parser.add_argument('--num-clusters', default=[32, 16, 8, 4], type=int, nargs='+',
                        help='Number of clusters for each segment, with [32, 16, 8, 4] for CIFAR and [64, 32, 16, 8] for ImageNet')
    parser.add_argument('--depth', default=[3, 3, 3, 2], type=int, nargs='+',
                        help='Depth of the encoder (CAST)')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--incremental-lr", default=None, type=float,
                        help="LR to use for incremental task (t > 0)")
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--incremental-warmup-lr', type=float, default=None, metavar='LR',
                        help='warmup learning rate (default: 1e-6) for task T > 0')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    # parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
    #                     help='Color jitter factor (default: 0.4)')
    # parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
    #                     help='Use AutoAugment policy. "v0" or "original". " + \
    #                          "(default: rand-m9-mstd0.5-inc1)'),
    # parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    # parser.add_argument('--train-interpolation', type=str, default='bicubic',
    #                     help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem", "old"')

    # Distillation parameters
    parser.add_argument('--auto-kd', default=False, action='store_true',
                        help='Balance kd factor as WA https://arxiv.org/abs/1911.07053')
    parser.add_argument('--kd', default=0., type=float)
    parser.add_argument('--distillation-tau', default=1.0, type=float,
                        help='Temperature for the KD')
    parser.add_argument('--resnet', default=False, action='store_true')
    parser.add_argument('--pod', default=None,  type=float)
    parser.add_argument('--pod-scales', default=[1], type=int, nargs='+')
    parser.add_argument('--pod-scaling', default=False,  action='store_true')

    # Dataset parameters
    parser.add_argument('--data-path', default='../data', type=str,
                        help='dataset path')
    # parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
    #                     type=str, help='Image Net dataset path')
    parser.add_argument('--data-set', type=str, default="iirc_cifar100",
                        choices=["incremental_cifar100", "iirc_cifar100", "incremental_imagenet_full",
                                 "incremental_imagenet_lite", "iirc_imagenet_full", "iirc_imagenet_lite",
                                 "iirc_imagenet_subset"])
    # parser.add_argument('--inat-category', default='name',
    #                     choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
    #                     type=str, help='semantic granularity')
    parser.add_argument('--output-dir', default='',
                        help='Dont use that')
    parser.add_argument('--output-basedir', default='./results/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # Continual Learning parameters
    # parser.add_argument("--initial-increment", default=50, type=int,
    #                     help="Base number of classes")
    # parser.add_argument("--increment", default=10, type=int,
    #                     help="Number of new classes per incremental task")
    # parser.add_argument('--class-order', default=None, type=int, nargs='+',
    #                     help='Class ordering, a list of class ids.')

    parser.add_argument("--eval-every", default=50, type=int,
                        help="Eval model every X epochs, if None only eval at the task end")
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Only do one batch per epoch')
    parser.add_argument('--retrain-scratch', default=False, action='store_true',
                        help='Retrain from scratch on all data after each step (JOINT).')
    parser.add_argument('--max-task', default=None, type=int,
                        help='Max task id to train on')
    parser.add_argument('--name', default='', help='Name to display for screen')
    parser.add_argument('--options', default=[], nargs='*')

    # DyTox related
    parser.add_argument('--dytox', action='store_true', default=False,
                        help='Enable super DyTox god mode.')
    parser.add_argument('--ind-clf', default='', choices=['1-1', '1-n', 'n-n', 'n-1'],
                        help='Independent classifier per task but predicting all seen classes')
    parser.add_argument('--joint-tokens', default=False, action='store_true',
                        help='Forward w/ all task tokens alltogether [Faster but not working as well, not sure why')

    # Diversity
    parser.add_argument('--head-div', default=0., type=float,
                        help='Use a divergent head to predict among new classes + 1 using last token')
    parser.add_argument('--head-div-mode', default=['tr', 'ft'], nargs='+', type=str,
                        help='Only do divergence during training (tr) and/or finetuning (ft).')

    # SAM-related parameters
    # SAM fails with Mixed Precision, so use --no-amp
    parser.add_argument('--sam-rho', default=0., type=float,
                        help='Rho parameters for Sharpness-Aware Minimization. Disabled if == 0.')
    # parser.add_argument('--sam-adaptive', default=False, action='store_true',
    #                     help='Adaptive version of SAM (more robust to rho)')
    # parser.add_argument('--sam-first', default='main', choices=['main', 'memory'],
    #                     help='Apply SAM first step on main or memory loader (need --sep-memory for the latter)')
    # parser.add_argument('--sam-second', default='main', choices=['main', 'memory'],
    #                     help='Apply SAM second step on main or memory loader (need --sep-memory for the latter)')
    # parser.add_argument('--sam-skip-first', default=False, action='store_true',
    #                     help='Dont use SAM for first task')
    # parser.add_argument('--sam-final', default=None, type=float,
    #                     help='Final value of rho is it is changed linearly per task.')
    # parser.add_argument('--sam-div', default='', type=str,
    #                     choices=['old_no_upd'],
    #                     help='SAM for diversity')
    # parser.add_argument('--sam-mode', default=['tr', 'ft'], nargs='+', type=str,
    #                     help='Only do SAM during training (tr) and/or finetuning (ft).')
    # parser.add_argument('--look-sam-k', default=0, type=int,
    #                     help='Apply look sam every K updates (see under review ICLR22)')
    # parser.add_argument('--look-sam-alpha', default=0.7, type=float,
    #                     help='Alpha factor of look sam to weight gradient reuse, 0 < alpha <= 1')

    # Parameters for creating the tasks in IIRC
    parser.add_argument('--tasks-configuration-id', type=int, default=0, choices=range(0, 10),
                        help="The task configuration id. Ignore for joint training")
    parser.add_argument('--complete-info', default=False, action='store_true',
                        help='use the complete information during training (a multi-label setting)')
    parser.add_argument('--incremental-joint', default=False, action='store_true',
                        help="keep all data from previous tasks, while updating their labels as per the observed "
                             "classes (use only with complete_info and without buffer)")
    parser.add_argument('--joint', default=False, action='store_true',
                        help="load all classes during the first task. This option ignores the tasks_configuration_id "
                             "(use only with complete_info and without buffer)")

    # The buffer parameters in IIRC
    parser.add_argument('--total-n-memories', type=int, default=-1,
                        help="The total replay buffer size, which is divided by the observed number of classes to get "
                             "the number of memories kept per task, note that the number of memories per task here is "
                             "not fixed but rather decreases as the number of tasks observed increases (with a minimum "
                             "of 1). If n_memories_per_class is set to a value greater than -1, the "
                             "n_memories_per_class is used instead.")
    parser.add_argument('--n-memories-per-class', type=int, default=20,
                        help="The number of samples to keep from each class, if set to -1, the total_n_memories "
                             "argument is used instead")
    parser.add_argument('--buffer-sampling-multiplier', type=float, default=1.0,
                        help="A multiplier for sampling from the buffer more/less times than the size of the buffer "
                             "(for example a multiplier of 2 samples from the buffer (with replacement) twice its size "
                             "per epoch)")
    # parser.add_argument('--memory_strength', type=float, default=1.0,
    #                     help="a weight to be multiplied by the loss from the buffer")
    parser.add_argument('--max-mems-pool-per-class', type=int, default=1e5,
                        help="Maximum size of the samples pool per class from which the buffer chooses the exemplars, "
                             "use -1 for choosing from the whole class samples.")

    # Rehearsal memory
    # parser.add_argument('--memory-size', default=2000, type=int,
    #                     help='Total memory size in number of stored (image, label).')
    parser.add_argument('--distributed-memory', default=False, action='store_true',
                        help='Use different rehearsal memory per process.')
    # parser.add_argument('--global-memory', default=False, action='store_false', dest='distributed_memory',
    #                     help='Use same rehearsal memory for all process.')
    parser.set_defaults(distributed_memory=True)
    # parser.add_argument('--oversample-memory', default=1, type=int,
    #                     help='Amount of time we repeat the same rehearsal.')
    parser.add_argument('--oversample-memory-ft', default=1, type=int,
                        help='Amount of time we repeat the same rehearsal for finetuning, only for old classes not new classes.')
    # parser.add_argument('--rehearsal-test-trsf', default=False, action='store_true',
    #                     help='Extract features without data augmentation.')
    # parser.add_argument('--rehearsal-modes', default=1, type=int,
    #                     help='Select N on a single gpu, but with mem_size/N.')
    # parser.add_argument('--fixed-memory', default=False, action='store_true',
    #                     help='Dont fully use memory when no all classes are seen as in Hou et al. 2019')
    parser.add_argument('--rehearsal', default="random",
                        choices=[
                            'random',
                            'closest_token', 'closest_all',
                            'icarl_token', 'icarl_all',
                            'furthest_token', 'furthest_all'
                        ],
                        help='Method to herd sample for rehearsal.')
    parser.add_argument('--sep-memory', default=False, action='store_true',
                        help='Dont merge memory w/ task dataset but keep it alongside')
    # parser.add_argument('--replay-memory', default=0, type=int,
    #                     help='Replay memory according to Guido rule [NEED DOC]')

    # Finetuning
    parser.add_argument('--finetuning', default='', choices=['balanced', ''],
                        help='Whether to do a finetuning after each incremental task. Backbone are frozen.')
    parser.add_argument('--finetuning-epochs', default=30, type=int,
                        help='Number of epochs to spend in finetuning.')
    parser.add_argument('--finetuning-lr', default=5e-5, type=float,
                        help='LR during finetuning, will be kept constant.')
    parser.add_argument('--finetuning-teacher', default=False, action='store_true',
                        help='Use teacher/old model during finetuning for all kd related.')
    parser.add_argument('--finetuning-resetclf', default=False, action='store_true',
                        help='Reset classifier before finetuning phase (similar to GDumb/DER).')
    parser.add_argument('--only-ft', default=False, action='store_true',
                        help='Only train on FT data')
    parser.add_argument('--ft-no-sampling', default=False, action='store_true',
                        help='Dont use particular sampling for the finetuning phase.')

    # What to freeze
    parser.add_argument('--freeze-task', default=[], nargs="*", type=str,
                        help='What to freeze before every incremental task (t > 0).')
    parser.add_argument('--freeze-ft', default=[], nargs="*", type=str,
                        help='What to freeze before every finetuning (t > 0).')
    parser.add_argument('--freeze-eval', default=False, action='store_true',
                        help='Frozen layers are put in eval. Important for stoch depth')

    # Convit - CaiT
    parser.add_argument('--local-up-to-layer', default=10, type=int,
                        help='number of GPSA layers')
    parser.add_argument('--locality-strength', default=1., type=float,
                        help='Determines how focused each head is around its attention center')
    parser.add_argument('--class-attention', default=False, action='store_true',
                        help='Freeeze and Process the class token as done in CaiT')


    # Classification
    parser.add_argument('--bce-loss', default=False, action='store_true')

    # distributed training parameters
    parser.add_argument('--local-rank', default=None, type=int)
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    # Resuming
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-task', default=0, type=int, help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, help='resume from checkpoint')
    parser.add_argument('--save-every-epoch', default=None, type=int)

    parser.add_argument('--validation', default=0.0, type=float,
                        help='Use % of the training set as val, replacing the test.')

    return parser


def main(args):
    print(args)

    use_distillation = args.auto_kd
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # -------------------------------
    essential_transforms_fn, augmentation_transforms_fn = get_transforms(args.data_set)
    lifelong_datasets, tasks, class_names_to_idx = get_lifelong_datasets(args.data_set,
                              dataset_root=args.data_path,
                              tasks_configuration_id=args.tasks_configuration_id,
                              essential_transforms_fn=essential_transforms_fn,
                              augmentation_transforms_fn=augmentation_transforms_fn, cache_images=False,
                              joint=args.joint, n_segments=args.n_segments)
    nb_tasks = len(tasks)
    args.nb_tasks = nb_tasks
    args.increment_per_task = [len(task) for task in tasks]
    args.initial_increment = args.increment_per_task[0]
    args.increment = args.increment_per_task[-1]
    nb_total_classes = len(class_names_to_idx)
    args.nb_classes = nb_total_classes

    if args.complete_info:
        for lifelong_dataset in lifelong_datasets.values():
            lifelong_dataset.enable_complete_information_mode()
    # -------------------------------

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None

    model = factory_iirc_hcast.get_backbone(args)
    model.head = Classifier(
        model.embed_dim, nb_total_classes, args.initial_increment,
        args.increment, nb_tasks
    )
    model.to(device)
    # model will be on multiple GPUs, while model_without_ddp on a single GPU, but
    # it's actually the same model.
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)

    log_store = None
    # Start the logging process on disk ----------------------------------------
    if args.name:
        args.output_dir = os.path.join(
            args.output_basedir,
            f"{datetime.datetime.now().strftime('%y-%m-%d')}_{args.data_set}_{args.name}"
        )

        log_path = os.path.join(args.output_dir, f"logs.json") # 仅记录task end任务结尾时刻的指标
        long_log_path = os.path.join(args.output_dir, f"long_logs.json") # 记录所有时间节点的指标

        if utils.is_main_process():
            os.system("echo '\ek{}\e\\'".format(args.name))
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, f"config.json"), 'w+') as f:
                config = vars(args)
                config["nb_parameters"] = n_parameters
                json.dump(config, f, indent=2)
            with open(log_path, 'w+') as f:
                pass  # touch
            with open(long_log_path, 'w+') as f:
                pass  # touch
    else:
        log_store = None
        log_path = long_log_path = None
    if args.output_dir and utils.is_main_process():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.distributed:
        torch.distributed.barrier()

    args.class_names_to_idx = class_names_to_idx

    print('number of params:', n_parameters)

    # 梯度缩放器，当作函数进行调用时将完成 对损失进行梯度计算-反向传播-（优化器）参数更新-缩放器更新 整个一系列操作
    loss_scaler = scaler.ContinualScaler(args.no_amp)

    if args.mixup > 0. or args.cutmix > 0.:
        criterion = SoftTargetCrossEntropy()
        if args.bce_loss:
            criterion = soft_bce_with_logits
    elif args.bce_loss:
        criterion = bce_with_logits
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    teacher_model = None

    output_dir = Path(args.output_dir)

    buffer_dir = None
    map_size = None
    # 初始化回放集memory/buffer
    if "imagenet" in args.data_set:
        if args.n_memories_per_class > 0:
            buffer_dir = output_dir
            map_size = int(args.n_memories_per_class * nb_total_classes * 1.4e6)
        elif args.total_n_memories > 0:
            buffer_dir = output_dir
            map_size = int(args.total_n_memories * 1.4e6)
    buffer = Buffer(args, buffer_dir, map_size, essential_transforms_fn, augmentation_transforms_fn)

    nb_classes = args.initial_increment # nb_classes统计已见类别数量
    base_lr = args.lr
    start_time = time.time()

    if args.debug:
        args.base_epochs = 1
        args.epochs = 1

    if args.finetuning and not args.resume: # 从头训练且不中断才能记录完整的结果
        # 若有微调，则额外保存训练后微调前的全面评估结果
        _ave_test_m_jaccard_dict = {"metric_name": "afterTRbeforeFT_ave_test_m_jaccard"}
        _ave_test_strict_acc_dict = {"metric_name": "afterTRbeforeFT_ave_test_strict_acc"}
    else:
        _ave_test_m_jaccard_dict = None
        _ave_test_strict_acc_dict = None

    ave_test_m_jaccard_dict = {"metric_name": "ave_test_m_jaccard"}
    ave_test_strict_acc_dict = {"metric_name": "ave_test_strict_acc"}

    # --------------------------------------------------------------------------
    # Begin of the task loop
    for task_id in range(nb_tasks):
        if args.incremental_joint:
            for lifelong_dataset in lifelong_datasets.values():
                lifelong_dataset.load_tasks_up_to(task_id)
        else:
            for lifelong_dataset in lifelong_datasets.values():
                lifelong_dataset.choose_task(task_id)  # choose the data of the current task (train, intask_valid, posttask_valid, test)
        assert lifelong_datasets['test'].complete_information_mode is True

        dataset_train = lifelong_datasets['train']

        if args.max_task == task_id:
            print(f"Stop training because of max task")
            break

        print(f"Starting task id {task_id}/{nb_tasks - 1}")

        loader_memory = None
        if task_id > 0:
            # dataset_memory = memory.get_dataset(lifelong_datasets['train'])
            # loader_memory = factory.InfiniteLoader(factory.get_train_loaders(
            #     dataset_memory, args,
            #     args.replay_memory if args.replay_memory > 0 else args.batch_size
            # ))
            if not args.sep_memory: # sep_memory默认为False，故运行该分支，即合并新样本和回放样本为一个数据集
                previous_size = len(lifelong_datasets['train'])
                bsm = args.buffer_sampling_multiplier
                task_train_data_with_buffer = TaskDataMergedWithBuffer(buffer, lifelong_datasets['train'], buffer_sampling_multiplier=bsm)
                print(f"{len(task_train_data_with_buffer) - previous_size} samples added from memory.")
                dataset_train = task_train_data_with_buffer

        #     if args.only_ft: # 默认不运行
        #         lifelong_datasets['train'] = get_finetuning_dataset(lifelong_datasets['train'], buffer, 'balanced')

        # ----------------------------------------------------------------------
        # Initializing teacher model from previous task
        if use_distillation and task_id > 0:
            teacher_model = copy.deepcopy(model_without_ddp)
            teacher_model.freeze(['all']) # 用于蒸馏的旧模型要冻结所有参数
            teacher_model.eval()
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # Ensembling
        if args.dytox:
            # 在每个任务训练前 更新（task0前则初始化）模型，对于dytox方法，更新涉及：
            # 扩展已见类别数量，添加新task token，添加新任务分类头head，创建新（覆盖旧）多样性分类头head_div
            model_without_ddp = factory_iirc_hcast.update_dytox(model_without_ddp, task_id, args)
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # Adding new parameters to handle the new classes
        print("Adding new parameters")
        if task_id > 0 and not args.dytox:
            model_without_ddp.head.add_classes()

        if task_id > 0:
            # freeze_task: [old_task_tokens, old_heads]，即训练新任务时会冻结所有旧任务的task tokens和分类头heads
            model_without_ddp.freeze(args.freeze_task)
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # Learning rate and optimizer
        if task_id > 0 and args.incremental_batch_size: # CIFAR数据集上bs和ibs均为128
            args.batch_size = args.incremental_batch_size

        if args.incremental_lr is not None and task_id > 0: # CIFAR数据集上原lr和ilr均为0.0005，由于bs为128，单机下真实lr其实为0.0005/4=1.25e-4
            linear_scaled_lr = args.incremental_lr * args.batch_size * utils.get_world_size() / 512.0
        else:
            linear_scaled_lr = base_lr * args.batch_size * utils.get_world_size() / 512.0

        args.lr = linear_scaled_lr
        optimizer = create_optimizer(args, model_without_ddp)
        lr_scheduler, _ = create_scheduler(args, optimizer)
        # ----------------------------------------------------------------------

        if mixup_active: # dytox方法中为False
            mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing,
                num_classes=nb_classes,
                loader_memory=loader_memory
            )

        skipped_task = False
        initial_epoch = epoch = 0

        # 如果保存的最新断点对应的任务已经训练微调完，那么就指定start_task为下一个任务，否则该任务会被从头学
        # （但是，由于同任务内不同阶段的ckpt是同名覆盖的，所以不知道能不能区分出该ckpt是训练+微调结束的最终版？）
        if args.resume and args.start_task > task_id:
            # 中断后恢复训练的策略不是直接从断点任务继续，而是依旧要从task0进入任务循环遍历，
            # 若判断任务已学（即start_task>task_id）则令skipped_task为True，跳过epochs循环遍历
            # 当前代码下args.start_epoch参数其实无效，即断点若在任务训练中期，从该点恢复训练其实是从头学该任务
            utils.load_first_task_model(model_without_ddp, buffer, lifelong_datasets, loss_scaler, task_id, args)

            print("Skipping first task") # 此处虽称first task，但可以是已学过的需要跳过的任一任务
            epochs = 0 # 总训练epochs数为0，所以虽加载task0模型但是跳过训练
            train_stats = {"task_skipped": str(task_id)}
            skipped_task = True
        elif args.base_epochs is not None and task_id == 0:
            epochs = args.base_epochs
        else:
            epochs = args.epochs

        if args.distributed:
            del model
            model = torch.nn.parallel.DistributedDataParallel(
                model_without_ddp, device_ids=[args.gpu], find_unused_parameters=True)
            torch.distributed.barrier()
        else:
            model = model_without_ddp

        model_without_ddp.nb_epochs = epochs

        # Init SAM, for DyTox++ (see appendix) ---------------------------------
        sam = None
        # if args.sam_rho > 0. and 'tr' in args.sam_mode and ((task_id > 0 and args.sam_skip_first) or not args.sam_skip_first):
        #     if args.sam_final is not None:
        #         sam_step = (args.sam_final - args.sam_rho) / nb_tasks
        #         sam_rho = args.sam_rho + task_id * sam_step
        #     else:
        #         sam_rho = args.sam_rho
        #
        #     print(f'Initialize SAM with rho={sam_rho}')
        #     sam = SAM(
        #         optimizer, model_without_ddp,
        #         rho=sam_rho, adaptive=args.sam_adaptive,
        #         div=args.sam_div,
        #         use_look_sam=args.look_sam_k > 0, look_sam_alpha=args.look_sam_alpha
        #     )
        # ----------------------------------------------------------------------

        print(f"Start training for {epochs-initial_epoch} epochs")
        for epoch in range(initial_epoch, epochs):
            train_stats = train_one_epoch(
                model, criterion, dataset_train,
                optimizer, device, epoch, task_id, loss_scaler,
                args.clip_grad, mixup_fn,
                debug=args.debug,
                args=args,
                teacher_model=teacher_model,
                model_without_ddp=model_without_ddp,
                sam=sam,
                loader_memory=loader_memory,
                pod=args.pod if task_id > 0 else None, pod_scales=args.pod_scales,
                finetuning=False
            )
            # 每个epoch的train_stats均记录，以监控训练进度
            log_stats = {'task': task_id,
                         'phase': 'TR',
                         'epoch': epoch,
                         'n_parameters': n_parameters,
                         **train_stats
                         }
            if long_log_path is not None and utils.is_main_process():
                with open(long_log_path, 'a+') as f:
                    f.write(json.dumps({**log_stats, **model_without_ddp.epoch_log()}) + '\n')
            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            # If using the lmdb database, close it and open a new environment to kill active readers
            buffer.reset_lmdb_database()

            lr_scheduler.step(epoch)

            # 训练时期的评估，每eval_every epochs阶段性评估，若有微调还要在训练后微调前即刻进行一次评估
            if args.eval_every and epoch % args.eval_every  == 0:
                # loader_val，即使用测试（/验证）集数据进行评估
                # eval_and_log方法中实现了评估前更新本任务断点文件ckpt！！！
                eval_and_log_in(
                    args, output_dir, model, model_without_ddp, optimizer, lr_scheduler,
                    epoch, task_id, loss_scaler, n_parameters, device, lifelong_datasets['test'],
                    train_stats, None, long_log_path,
                    model_without_ddp.epoch_log(), phase=0)
            if args.finetuning and epoch == epochs - 1 and task_id > 0:
                eval_and_log_end(
                    args, output_dir, model, model_without_ddp, optimizer, lr_scheduler,
                    epoch, task_id, loss_scaler, n_parameters, device,
                    _ave_test_m_jaccard_dict, _ave_test_strict_acc_dict,
                    lifelong_datasets['test'], train_stats, None, log_path,
                    model_without_ddp.epoch_log(), t_endpoint=0)


        # 任务训练后更新回放集memory
        if args.distributed_memory and not skipped_task:
                # expand buffer
                buffer.update_buffer_new_task(
                    lifelong_datasets["train"], args, model=model, batch_size=args.batch_size, device=device
                )

        # ----------------------------------------------------------------------
        # FINETUNING
        # ----------------------------------------------------------------------
        # Init SAM, for DyTox++ (see appendix) ---------------------------------
        sam = None
        # if args.sam_rho > 0. and 'ft' in args.sam_mode and ((task_id > 0 and args.sam_skip_first) or not args.sam_skip_first):
        #     if args.sam_final is not None:
        #         sam_step = (args.sam_final - args.sam_rho) / nb_tasks
        #         sam_rho = args.sam_rho + task_id * sam_step
        #     else:
        #         sam_rho = args.sam_rho
        #
        #     print(f'Initialize SAM with rho={sam_rho}')
        #     sam = SAM(
        #         optimizer, model_without_ddp,
        #         rho=sam_rho, adaptive=args.sam_adaptive,
        #         div=args.sam_div,
        #         use_look_sam=args.look_sam_k > 0, look_sam_alpha=args.look_sam_alpha
        #     )
        # ----------------------------------------------------------------------

        # 微调的前提条件中 or的部分 是为什么？
        if args.finetuning and (task_id > 0 or nb_total_classes == args.initial_increment) and not skipped_task:
            # 微调时采用平衡的训练数据集，其实就是更新后的memory回放集的数据，并使用完整的测试（/验证）集做评估
            dataset_finetune = get_finetuning_dataset(lifelong_datasets['train'], buffer, args.finetuning, args.oversample_memory_ft, task_id)
            print(f'Finetuning phase of type {args.finetuning} with {len(dataset_finetune)} samples.')

            if args.finetuning_resetclf: # CIFAR默认不重设，故忽略该分支
                model_without_ddp.reset_classifier()

            model_without_ddp.freeze(args.freeze_ft) # CIFAR冻结backbone，即编码器SABs，微调只更新TAB、task tokens、heads

            if args.distributed:
                del model
                model = torch.nn.parallel.DistributedDataParallel(model_without_ddp, device_ids=[args.gpu], find_unused_parameters=True)
                torch.distributed.barrier()
            else:
                model = model_without_ddp

            model_without_ddp.begin_finetuning()

            args.lr  = args.finetuning_lr * args.batch_size * utils.get_world_size() / 512.0
            optimizer = create_optimizer(args, model_without_ddp)
            for epoch in range(args.finetuning_epochs):
                # CIFAR设置args.finetuning_teacher为False，故微调时不计算和考虑蒸馏KD
                # 且由于args.head_div_mode为'tr'，故微调时不计算logits_div和div_loss。综上微调时仅计算bce分类损失
                train_stats = train_one_epoch(
                    model, criterion, dataset_finetune,
                    optimizer, device, epoch, task_id, loss_scaler,
                    args.clip_grad, mixup_fn,
                    debug=args.debug,
                    args=args,
                    teacher_model=teacher_model if args.finetuning_teacher else None,
                    model_without_ddp=model_without_ddp,
                    pod=args.pod if task_id > 0 else None, pod_scales=args.pod_scales,
                    finetuning=True
                )

                # If using the lmdb database, close it and open a new environment to kill active readers
                buffer.reset_lmdb_database()

                # finetuning微调时期评估，每10 epochs阶段性评估，微调结束即刻也进行一次评估
                # 评估前会保存本任务断点文件ckpt
                # if epoch % 10 == 0:
                if epoch % 2 == 0:
                    eval_and_log_in(
                        args, output_dir, model, model_without_ddp, optimizer, lr_scheduler,
                        epoch, task_id, loss_scaler, n_parameters, device,
                        lifelong_datasets['test'], train_stats, None, long_log_path,
                        model_without_ddp.epoch_log(), phase=1)

            model_without_ddp.end_finetuning()

        # 任务后（本任务的训练+微调全部结束后）评估，记录accuracy_list
        # 评估前会保存本任务断点文件ckpt，本次保存的即为该任务对应的最终文件
        eval_and_log_end(
            args, output_dir, model, model_without_ddp, optimizer, lr_scheduler,
            epoch, task_id, loss_scaler, n_parameters, device,
            ave_test_m_jaccard_dict, ave_test_strict_acc_dict,
            lifelong_datasets['test'], train_stats, log_store, log_path,
            model_without_ddp.epoch_log(), skipped_task, t_endpoint=1)

        # 评估前保存（更新/覆盖）模型断点文件ckpt
        # 同一任务内不同阶段保存的断点文件同名，所以会覆盖
        if args.output_dir:
            if os.path.isdir(args.resume):
                checkpoint_paths = [os.path.join(args.resume, f'checkpoint_{task_id}.pth')]
            else:
                checkpoint_paths = [output_dir / f'checkpoint_{task_id}.pth']
            for checkpoint_path in checkpoint_paths:
                if skipped_task:
                    continue

                utils.save_on_master({
                    'datasets': {dataset_type: lifelong_datasets[dataset_type].dataset_state_dict() for dataset_type in
                                 lifelong_datasets.keys()},
                    'buffer': buffer.buffer_state_dict(),
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'task_id': task_id,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

        nb_classes += args.increment
    # --------------------------------------------------------------------------

    if _ave_test_m_jaccard_dict is not None and _ave_test_strict_acc_dict is not None:  # 微调情况下，只有从头训练且不中断，才能记录所有任务上的完整的ave_test_...指标集合，从而计算Ave
        # task0无需微调，所以两指标在两个endpoint上的值相同
        _ave_test_m_jaccard_dict.update({"task0": ave_test_m_jaccard_dict["task0"]})
        _ave_test_strict_acc_dict.update({"task0": ave_test_strict_acc_dict["task0"]})
        assert len(_ave_test_m_jaccard_dict) == nb_tasks + 1
        assert len(_ave_test_strict_acc_dict) == nb_tasks + 1
        _ave_test_m_jaccard_dict.update(
            {"Ave": sum(list(_ave_test_m_jaccard_dict.values())[1:]) / (len(_ave_test_m_jaccard_dict) - 1)})
        _ave_test_strict_acc_dict.update(
            {"Ave": sum(list(_ave_test_strict_acc_dict.values())[1:]) / (len(_ave_test_strict_acc_dict) - 1)})
        if log_path is not None and utils.is_main_process():
            with open(log_path, 'a+') as f:
                f.write(json.dumps(_ave_test_m_jaccard_dict) + '\n')
                f.write(json.dumps(_ave_test_strict_acc_dict) + '\n')
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps({**_ave_test_m_jaccard_dict, **_ave_test_strict_acc_dict}) + "\n")
        print(_ave_test_m_jaccard_dict)
        print(_ave_test_strict_acc_dict)

    assert len(ave_test_m_jaccard_dict) == nb_tasks + 1
    assert len(ave_test_strict_acc_dict) == nb_tasks + 1
    ave_test_m_jaccard_dict.update(
        {"Ave": sum(list(ave_test_m_jaccard_dict.values())[1:]) / (len(ave_test_m_jaccard_dict) - 1)})
    ave_test_strict_acc_dict.update(
        {"Ave": sum(list(ave_test_strict_acc_dict.values())[1:]) / (len(ave_test_strict_acc_dict) - 1)})
    if log_path is not None and utils.is_main_process():
        with open(log_path, 'a+') as f:
            f.write(json.dumps(ave_test_m_jaccard_dict) + '\n')
            f.write(json.dumps(ave_test_strict_acc_dict) + '\n')
    if args.output_dir and utils.is_main_process():
        with (output_dir / "log.txt").open("a") as f:
            f.write(json.dumps({**ave_test_m_jaccard_dict, **ave_test_strict_acc_dict}) + "\n")
    print(ave_test_m_jaccard_dict)
    print(ave_test_strict_acc_dict)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def load_options(args, options):
    varargs = vars(args)

    name = []
    for o in options:
        with open(o) as f:
            new_opts = yaml.safe_load(f)

        for k, v in new_opts.items():
            if k not in varargs:
                raise ValueError(f'Option {k}={v} doesnt exist!')
        varargs.update(new_opts)
        name.append(o.split("/")[-1].replace('.yaml', ''))

    return '_'.join(name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DyTox training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    utils.init_distributed_mode(args)

    if args.options:
        name = load_options(args, args.options)
        if os.path.isdir(args.resume):
            run_id = int(args.resume.rstrip('/').split('_')[-1])
        else:
            run_id = random.randint(0, 100000)
        print(f'Run ID: {run_id}')
        name = f'{name}_{run_id}'

        if not args.name:
            args.name = name

    main(args)
