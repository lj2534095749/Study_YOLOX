#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn

from yolox.core import Trainer, launch
from yolox.exp import get_exp
from yolox.utils import configure_nccl, configure_omp, get_num_devices


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")

    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # ---------------------------------------------------------------------#
    #   distributed     用于指定是否使用单机多卡分布式运行
    #                   终端指令仅支持Ubuntu。CUDA_VISIBLE_DEVICES用于在Ubuntu下指定显卡。
    #                   Windows系统下默认使用DP模式调用所有显卡，不支持DDP。
    #   DP模式：
    #       设置            distributed = False
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP模式：
    #       设置            distributed = True
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    # ---------------------------------------------------------------------#
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--dist-url", default=None, type=str, help="url used to set up distributed training", )

    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("-d", "--devices", default=None, type=int, help="device for training")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="plz input your experiment description file", )
    parser.add_argument("--resume", default=False, action="store_true", help="resume training")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument("-e", "--start_epoch", default=None, type=int, help="resume training start epoch")
    parser.add_argument("--num_machines", default=1, type=int, help="num of node for training")
    parser.add_argument("--machine_rank", default=0, type=int, help="node rank for multi-node training")

    # ---------------------------------------------------------------------#
    #   fp16        是否使用混合精度训练
    #               可减少约一半的显存、需要pytorch1.7.1以上
    # ---------------------------------------------------------------------#
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true",
                        help="Adopting mix precision training.", )

    parser.add_argument("--cache", dest="cache", default=False, action="store_true",
                        help="Caching imgs to RAM for fast training.", )
    parser.add_argument("-o", "--occupy", dest="occupy", default=False, action="store_true",
                        help="occupy GPU memory first for training.", )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER, )
    return parser


@logger.catch
def main(exp, args):
    # ------------------------------------------------------ #
    # 是否进行seed training训练
    # 默认不进行
    # ------------------------------------------------------ #
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    configure_nccl()
    configure_omp()
    # ------------------------------------------------------ #
    # 代码释意：通过如上设置让内置的cuDNN的auto - tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
    # 遵循准则：
    #   网络的输入数据维度或类型上变化不大，设置为true可以增加运行效率；
    #   如果网络的输入数据在每个iteration都变化的话，会导致cnDNN每次都会去寻找一遍最优配置，这样反而会降低运行效率;
    # Note:
    #   cuDNN是英伟达专门为深度神经网络所开发出来的GPU加速库，针对卷积、池化等等常见操作做了非常多的底层优化，比一般的GPU程序要快很多。
    #   在使用cuDNN的时候，默认为False。
    #   设置为True将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。
    #   适用场景是网络结构固定，网络输入形状不变（即一般情况下都适用）。
    #   反之，如果卷积层的设置一直变化，将会导致程序不停地做优化，反而会耗费更多的时间。
    # ------------------------------------------------------ #
    cudnn.benchmark = True

    trainer = Trainer(exp, args)
    trainer.train()


if __name__ == "__main__":
    # ------------------------------------------------------ #
    # 设置参数
    # ------------------------------------------------------ #
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    # ------------------------------------------------------ #
    # 配置experiment_name
    # ------------------------------------------------------ #
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    # ------------------------------------------------------ #
    # 获取GPU个数
    # ------------------------------------------------------ #
    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()

    # ------------------------------------------------------ #
    # url to connect to for distributed training, including protocol
    # e.g. "tcp://127.0.0.1:8686".
    # Can be set to auto to automatically select a free port on localhost
    # ------------------------------------------------------ #
    dist_url = "auto" if args.dist_url is None else args.dist_url

    # ------------------------------------------------------ #
    # 根据num_machines判断是否进行多设备训练
    # 单设备（num_machines=1）时, 直接执行main方法
    # ------------------------------------------------------ #
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args),
    )
