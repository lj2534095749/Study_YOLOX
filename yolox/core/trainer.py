#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import datetime
import os
import time
from loguru import logger

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from yolox.data import DataPrefetcher
from yolox.utils import (
    MeterBuffer,
    ModelEMA,
    all_reduce_norm,
    get_local_rank,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    is_parallel,
    load_ckpt,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize
)


class Trainer:
    def __init__(self, exp, args):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        self.args = args

        # training related attr
        self.max_epoch = exp.max_epoch  # 最大论数
        self.amp_training = args.fp16  # 是否使用混合精度训练
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)  # 是否使用混合精度训练
        self.is_distributed = get_world_size() > 1  # 用于指定是否使用单机多卡分布式运行
        self.rank = get_rank()  # 单设备值为0
        self.local_rank = get_local_rank()  # 单设备值为0
        self.device = "cuda:{}".format(self.local_rank)  # 使用的设备
        self.use_model_ema = exp.ema  # 指数滑动平均

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32  # 输入数据类型
        self.input_size = exp.input_size  # 输入数据大小
        self.best_ap = 0

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    def train_one_iter(self):
        # 当前iter开始时间
        iter_start_time = time.time()

        # Prefetcher could speedup your pytorch dataloader
        inps, targets = self.prefetcher.next()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        # 取消targets的梯度
        targets.requires_grad = False
        # 设置inputs的大小为tsize，并使targets进行适配
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        # 数据获取结束时间
        data_end_time = time.time()

        # 将inputs, targets装入模型，amp_training=true进行混合精度训练
        with torch.cuda.amp.autocast(enabled=self.amp_training):
            outputs = self.model(inps, targets)

        # 损失
        loss = outputs["total_loss"]
        # 告诉优化器把梯度属性中权重的梯度归零，否则pytorch会累积梯度
        self.optimizer.zero_grad()
        # 反向传播
        self.scaler.scale(loss).backward()
        # 应用优化器
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Update EMA parameters
        if self.use_model_ema:
            self.ema_model.update(self.model)

        # 更新学习率
        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        # 当前iter结束时间
        iter_end_time = time.time()

        # 更新meter
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )

    def before_train(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(self.local_rank)  # 设置使用的设备
        model = self.exp.get_model()  # 获取模型
        logger.info(
            "Model Summary: {}".format(get_model_info(model, self.exp.test_size))
        )  # 获取模型参数大小
        model.to(self.device)  # 将模型加载到使用的设备

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)  # 获取SGD优化器

        # value of epoch will be set in `resume_train`
        model = self.resume_train(model)  # 断点续训或加载模型参数

        # data related init
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs  # 设置不进行数据增强的开始轮数
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
            cache_img=self.args.cache,
        )

        logger.info("init prefetcher, this might take one minute or less...")
        # Prefetcher could speedup your pytorch dataloader
        self.prefetcher = DataPrefetcher(self.train_loader)

        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)  # 每个轮数epoch的最大步数iter

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )  # 学习率

        if self.args.occupy:
            occupy_mem(self.local_rank)  # pre-allocate gpu memory for training to avoid memory Fragmentation.

        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)  # 分布式训练

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)  # 指数滑动平均
            self.ema_model.updates = self.max_iter * self.start_epoch  # 断点续训时更新

        self.model = model
        self.model.train()  # 开始训练

        self.evaluator = self.exp.get_evaluator(
            batch_size=self.args.batch_size, is_distributed=self.is_distributed
        )
        # Tensorboard logger
        if self.rank == 0:
            self.tblogger = SummaryWriter(self.file_name)

        logger.info("Training start...")
        logger.info("\n{}".format(model))

    def after_train(self):
        logger.info(
            "Training of experiment is done and the best AP is {:.2f}".format(self.best_ap * 100)
        )

    def before_epoch(self):
        logger.info("---> start train epoch{}".format(self.epoch + 1))

        if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:  # 如果没有了数据增强
            logger.info("--->No mosaic aug now!")
            self.train_loader.close_mosaic()  # 关闭马赛克数据增强
            logger.info("--->Add additional L1 loss now!")
            if self.is_distributed:
                self.model.module.head.use_l1 = True
            else:
                self.model.head.use_l1 = True  # 使用L1 loss
            self.exp.eval_interval = 1
            if not self.no_aug:
                self.save_ckpt(ckpt_name="last_mosaic_epoch")  # 保存没有数据增强前最后一轮的训练参数

    def after_epoch(self):
        self.save_ckpt(ckpt_name="latest")  # 保存训练参数

        if (self.epoch + 1) % self.exp.eval_interval == 0:  # 每训练几轮就进行验证，并保存模型
            all_reduce_norm(self.model)
            self.evaluate_and_save_model()

    def before_iter(self):
        pass

    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) % self.exp.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            logger.info(
                "{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    gpu_mem_usage(),
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.input_size[0], eta_str))
            )
            self.meter.clear_meters()

        # random resizing
        if (self.progress_in_iter + 1) % 10 == 0:
            self.input_size = self.exp.random_resize(
                self.train_loader, self.epoch, self.rank, self.is_distributed
            )

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        """断点续训或加载模型参数
        # 如果开启断点续训，直接加载模型参数，并修改开始轮数
        # 否则，根据参数和模型是否匹配进行模型参数加载
        """
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt

            ckpt = torch.load(ckpt_file, map_location=self.device)
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            # resume the training states variables
            start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt["start_epoch"]
            )
            self.start_epoch = start_epoch
            logger.info(
                "loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.start_epoch
                )
            )  # noqa
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt  # 得到模型参数文件
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]  # 得到模型参数
                model = load_ckpt(model, ckpt)    # 加载模型参数
            self.start_epoch = 0

        return model

    def evaluate_and_save_model(self):
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module

        ap50_95, ap50, summary = self.exp.eval(
            evalmodel, self.evaluator, self.is_distributed
        )
        self.model.train()
        if self.rank == 0:
            self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
            self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
            logger.info("\n" + summary)
        synchronize()

        self.save_ckpt("last_epoch", ap50_95 > self.best_ap)
        self.best_ap = max(self.best_ap, ap50_95)

    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )
