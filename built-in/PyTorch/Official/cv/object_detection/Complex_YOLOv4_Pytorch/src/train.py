'''# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==========================================================================
'''
import time
import numpy as np
import sys
import random
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import torch
if torch.__version__ >= '1.8':
    import torch_npu

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from tqdm import tqdm
import apex
from apex import amp
try:
    from torch_npu.utils.profiler import Profile
except ImportError:
    print("Profile not in torch_npu.utils.profiler now... Auto Profile disabled.", flush=True)
    class Profile:
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            pass
        def end(self):
            pass
sys.path.append('./')

from data_process.kitti_dataloader import create_train_dataloader, create_val_dataloader
from models.model_utils import create_model, make_data_parallel, get_num_parameters
from utils.train_utils import create_optimizer, create_lr_scheduler, get_saved_state, save_checkpoint
from utils.train_utils import reduce_tensor, to_python_float, get_tensorboard_log
from utils.misc import AverageMeter, ProgressMeter
from utils.logger import Logger
from config.train_config import parse_train_configs
from evaluate import evaluate_mAP


def main():
    amp.register_float_function(torch, 'sigmoid')

    configs = parse_train_configs()
    # Re-produce results
    if configs.seed is not None:
        random.seed(configs.seed)
        np.random.seed(configs.seed)
        torch.manual_seed(configs.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if configs.dist_url == "env://" and configs.world_size == -1:
        configs.world_size = int(os.environ["WORLD_SIZE"])

    configs.distributed = configs.world_size > 1 or configs.multiprocessing_distributed

    if configs.multiprocessing_distributed:
        configs.world_size = configs.ngpus_per_node * configs.world_size

    device = torch.device('cpu' if configs.local_rank is None else 'npu:{}'.format(configs.local_rank))

    if configs.distributed:
        dist.init_process_group(backend=configs.dist_backend, init_method=configs.dist_url,
                                world_size=configs.world_size, rank=configs.local_rank)
        configs.subdivisions = max(int(64 / configs.batch_size / configs.ngpus_per_node), 1)
    else:
        configs.subdivisions = max(int(64 / configs.batch_size), 1)

    configs.is_master_node = (not configs.distributed) or (
            configs.distributed and (configs.local_rank % configs.ngpus_per_node == 0))

    if configs.is_master_node:
        if not os.path.isdir(configs.checkpoints_dir):
            os.makedirs(configs.checkpoints_dir)
        if not os.path.isdir(configs.logs_dir):
            os.makedirs(configs.logs_dir)
        logger = Logger(configs.logs_dir, configs.saved_fn)
        logger.info('>>> Created a new logger')
        logger.info('>>> configs: {}'.format(configs))
    else:
        logger = None

    # model
    model = create_model(configs)

    if configs.local_rank is not None:
        torch.npu.set_device(configs.local_rank)
        model.npu(configs.local_rank)
    else:
        model.npu()

    # load weight from a checkpoint
    if configs.pretrained_path is not None:
        if not os.path.isfile(configs.pretrained_path):
            raise FileNotFoundError("=> no checkpoint found at '{}'".format(configs.pretrained_path))
        model.load_state_dict(torch.load(configs.pretrained_path, map_location=device))
        if logger is not None:
            logger.info('loaded pretrained model at {}'.format(configs.pretrained_path))

    # resume weights of model from a checkpoint
    if configs.resume_path is not None:
        if not os.path.isfile(configs.resume_path):
            raise FileNotFoundError("=> no checkpoint found at '{}'".format(configs.resume_path))
        model.load_state_dict(torch.load(configs.resume_path, map_location=device))
        if logger is not None:
            logger.info('resume training model from checkpoint {}'.format(configs.resume_path))

    # Make sure to create optimizer after moving the model to npu
    optimizer = create_optimizer(configs, model)
    lr_scheduler = create_lr_scheduler(optimizer, configs)
    configs.step_lr_in_epoch = True if configs.lr_type in ['multi_step'] else False

    # Amp
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", user_cast_preferred=True, combine_grad=True)

    # Data Parallel
    model = make_data_parallel(model, configs)

    # resume optimizer, lr_scheduler from a checkpoint
    if configs.resume_path is not None:
        utils_path = configs.resume_path.replace('Model_', 'Utils_')
        if not os.path.isfile(utils_path):
            raise FileNotFoundError("=> no checkpoint found at '{}'".format(utils_path))
        utils_state_dict = torch.load(utils_path, map_location='npu:{}'.format(configs.local_rank))
        optimizer.load_state_dict(utils_state_dict['optimizer'])
        lr_scheduler.load_state_dict(utils_state_dict['lr_scheduler'])
        configs.start_epoch = utils_state_dict['epoch'] + 1

    if configs.is_master_node:
        num_parameters = get_num_parameters(model)
        logger.info('number of trained parameters of the model: {}'.format(num_parameters))
        if logger is not None :
            logger.info(">>> Loading dataset & getting dataloader...")
    # Create dataloader
    train_dataloader, train_sampler = create_train_dataloader(configs)
    if logger is not None and configs.is_master_node:
        logger.info('number of batches in training set: {}'.format(len(train_dataloader)))

    if configs.prof:
        profiling(train_dataloader, model, optimizer, lr_scheduler, 0, configs, logger, device)

    if configs.evaluate:
        val_dataloader = create_val_dataloader(configs)
        precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs, None)
        print('Evaluate - precision: {}, recall: {}, AP: {}, f1: {}, ap_class: {}'.format(precision, recall, AP, f1,
                                                                                          ap_class))
        print('mAP {}'.format(AP.mean()))
        return

    best = 0
    for epoch in range(configs.start_epoch, configs.num_epochs + 1):
        if configs.is_master_node:
            if logger is not None:
                logger.info('{}'.format('*-' * 40))
                logger.info('{} {}/{} {}'.format('=' * 35, epoch, configs.num_epochs, '=' * 35))
                logger.info('{}'.format('*-' * 40))
                logger.info('>>> Epoch: [{}/{}]'.format(epoch, configs.num_epochs))

        if configs.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_one_epoch(train_dataloader, model, optimizer, lr_scheduler, epoch, configs, logger, device)

        # Save checkpoint
        if epoch % configs.checkpoint_freq == 0:
            if configs.is_master_node:
                model_state_dict, utils_state_dict = get_saved_state(model, optimizer, lr_scheduler, epoch, configs)
                save_checkpoint(configs.checkpoints_dir, configs.saved_fn, model_state_dict, utils_state_dict, epoch)
            if epoch > 240:
                if not configs.no_val:
                    val_dataloader = create_val_dataloader(configs)
                    print('number of batches in val_dataloader: {}'.format(len(val_dataloader)))
                    precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs, logger, device)
                    mAP = AP.mean()
                    val_metrics_dict = {
                        'precision': precision.mean(),
                        'recall': recall.mean(),
                        'AP': mAP,
                        'f1': f1.mean(),
                        'ap_class': ap_class.mean()
                    }
                    print(val_metrics_dict)
                # Save best checkpoint
                if mAP > best:
                    if configs.is_master_node:
                        best = mAP
                        model_state_dict, utils_state_dict = get_saved_state(model, optimizer, lr_scheduler, epoch, configs)
                        save_checkpoint(configs.checkpoints_dir, configs.saved_fn, model_state_dict, utils_state_dict, 'best')

        if not configs.step_lr_in_epoch:
            lr_scheduler.step()

def train_one_epoch(train_dataloader, model, optimizer, lr_scheduler, epoch, configs, logger, device):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    progress = ProgressMeter(len(train_dataloader), [batch_time, data_time, losses],
                             prefix="Train - Epoch: [{}/{}]".format(epoch, configs.num_epochs))

    num_iters_per_epoch = len(train_dataloader)

    # switch to train mode
    model.train()
    torch.npu.synchronize()
    start_time = time.time()
    profiler = Profile(start_step=int(os.getenv("PROFILE_START_STEP", 10)),
                       profile_type=os.getenv("PROFILE_TYPE"))
    for batch_idx, batch_data in enumerate(tqdm(train_dataloader)):
        if configs.perf and batch_idx == 400:
            break
        torch.npu.synchronize()
        data_time.update(time.time() - start_time)

        _, imgs, targets = batch_data
        global_step = num_iters_per_epoch * (epoch - 1) + batch_idx + 1

        batch_size = imgs.size(0)

        targets = targets.to(device, non_blocking=True)
        skip = torch.tensor(0.).to(device)
        if targets is None or targets.size(0) == 0:
            skip = torch.tensor(1.).to(device)
        if configs.distributed:
            torch.distributed.all_reduce(skip)
        if skip > 0:
            continue
        imgs = imgs.to(device, non_blocking=True)
        profiler.start()
        total_loss, outputs = model(imgs, targets)

        # For torch.nn.DataParallel case
        if (not configs.distributed) and (configs.local_rank is None):
            total_loss = torch.mean(total_loss)

        # compute gradient and perform backpropagation
        with amp.scale_loss(total_loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        if global_step % configs.subdivisions == 0:
            optimizer.step()
            # Adjust learning rate
            if configs.step_lr_in_epoch:
                lr_scheduler.step()
            # zero the parameter gradients
            optimizer.zero_grad()
        profiler.end()
        losses.update(to_python_float(total_loss.data), batch_size)

        # measure elapsed time
        torch.npu.synchronize()
        if batch_idx == 5:
            batch_time = AverageMeter('Time', ':6.3f')
            progress = ProgressMeter(len(train_dataloader), [batch_time, data_time, losses],
                             prefix="Train - Epoch: [{}/{}]".format(epoch, configs.num_epochs))

        batch_time.update(time.time() - start_time)
        start_time = time.time()
        # Log message
        if logger is not None and configs.is_master_node:
            if (global_step % configs.print_freq) == 0:
                logger.info(progress.get_message(batch_idx))


def profiling(train_dataloader, model, optimizer, lr_scheduler, epoch, configs, logger, device):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    progress = ProgressMeter(len(train_dataloader), [batch_time, data_time, losses],
                             prefix="Train - Epoch: [{}/{}]".format(epoch, configs.num_epochs))

    num_iters_per_epoch = len(train_dataloader)

    # switch to train mode
    model.train()
    torch.npu.synchronize()
    start_time = time.time()
    for batch_idx, batch_data in enumerate(tqdm(train_dataloader)):
        with torch.autograd.profiler.profile(use_npu=True) as prof:
            data_time.update(time.time() - start_time)
            _, imgs, targets = batch_data
            global_step = num_iters_per_epoch * (epoch - 1) + batch_idx + 1

            batch_size = imgs.size(0)

            targets = targets.to(device, non_blocking=True)
            imgs = imgs.to(device, non_blocking=True)
            total_loss, outputs = model(imgs, targets)

            # For torch.nn.DataParallel case
            if (not configs.distributed) and (configs.local_rank is None):
                total_loss = torch.mean(total_loss)

            # compute gradient and perform backpropagation
            # total_loss.backward()
            with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if global_step % configs.subdivisions == 0:
                optimizer.step()
                # Adjust learning rate
                if configs.step_lr_in_epoch:
                    lr_scheduler.step()
                # zero the parameter gradients
                optimizer.zero_grad()

            if configs.distributed:
                reduced_loss = reduce_tensor(total_loss.data, configs.world_size)
            else:
                reduced_loss = total_loss.data
            losses.update(to_python_float(reduced_loss), batch_size)
            # measure elapsed time
            torch.npu.synchronize()
            batch_time.update(time.time() - start_time)

            # Log message
            if logger is not None and configs.is_master_node:
                if (global_step % configs.print_freq) == 0:
                    logger.info(progress.get_message(batch_idx))

            torch.npu.synchronize()
            start_time = time.time()
        if batch_idx == 30:
            prof.export_chrome_trace("com_yolo.prof") # "output.prof"为输出文件地址
            print(prof.key_averages().table(sort_by="self_cpu_time_total"))
            sys.exit(0)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        try:
            cleanup()
            sys.exit(0)
        except SystemExit:
            os._exit(0)
