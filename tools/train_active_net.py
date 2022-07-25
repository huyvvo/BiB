#------------------------------------------------------------------------------
# Code adapted from https://github.com/NVlabs/wetectron
# by Huy V. Vo and Oriane Simeoni                              
# INRIA, Valeo.ai                                                             
#------------------------------------------------------------------------------

# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# --------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
from wetectron.utils.env import setup_environment  # noqa F401 isort:skip
import wetectron
print(wetectron.__file__)
import argparse
import os
import shutil
import random
import warnings
import logging
import numpy as np
import pickle
import torch
from wetectron.config import cfg
from wetectron.data import make_data_loader
from wetectron.solver import make_lr_scheduler, make_lr_cdb_scheduler
from wetectron.solver import make_optimizer, make_cdb_optimizer
from wetectron.engine.inference import inference
from wetectron.engine.trainer import do_train, do_train_cdb, do_train_cdb_with_iter_size
from wetectron.modeling.detector import build_detection_model
from wetectron.utils.checkpoint import DetectronCheckpointer
from wetectron.utils.collect_env import collect_env_info
from wetectron.utils.comm import synchronize, get_rank
from wetectron.utils.imports import import_file
from wetectron.utils.logger import setup_logger
from wetectron.utils.miscellaneous import mkdir, save_config, seed_all_rng, get_run_name, copy_source_code
from wetectron.utils.metric_logger import (MetricLogger, TensorboardLogger)
from wetectron.modeling.cdb import ConvConcreteDB

import torch.multiprocessing

try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')

def train(cfg, local_rank, distributed, use_tensorboard=False, retrain=False, load_optimizer=False):
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False, find_unused_parameters=True,
        )

    arguments = {"iteration": 0, "iter_size": cfg.SOLVER.ITER_SIZE}
    output_dir = cfg.OUTPUT_DIR
    save_to_disk = get_rank() == 0

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    trainlog_period = cfg.SOLVER.TRAINLOG_PERIOD
    ckpt_output_dir = os.path.join(output_dir, 'ckpt')
    mkdir(ckpt_output_dir)
    checkpointer = DetectronCheckpointer(
        cfg, model , optimizer, scheduler, ckpt_output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT, model_only=not retrain)
    if "optimizer" in extra_checkpoint_data and load_optimizer:
        extra_checkpoint_data = checkpointer.load_optimizer(extra_checkpoint_data)

    if retrain:
        arguments["iteration"] = extra_checkpoint_data["iteration"]
        if cfg.SOLVER.STEPS != scheduler.milestones:
            print(f'Updating scheduler.milestones to {cfg.SOLVER.STEPS}')
            scheduler.milestones = cfg.SOLVER.STEPS

        if cfg.SOLVER.ITER_SIZE != extra_checkpoint_data['iter_size']:
            raise Exception(f'iter_size in cfg and the loaded checkpoint are different')
    print("DetectronCheckpointer")
    
    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    if use_tensorboard:
        meters = TensorboardLogger(
            log_dir=os.path.join(cfg['OUTPUT_DIR'], 'log/'),
            start_iter=arguments['iteration'],
            delimiter="  ")
    else:
        meters = MetricLogger(delimiter="  ")

    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        meters,
        trainlog_period,
        cfg
    )
    synchronize()
    return model


def train_cdb(cfg, local_rank, distributed, use_tensorboard=False, retrain=False, load_optimizer=False):
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    model_cdb = ConvConcreteDB(cfg, model.backbone.out_channels)
    model_cdb.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)
    optimizer_cdb = make_cdb_optimizer(cfg, model_cdb)
    scheduler_cdb = make_lr_cdb_scheduler(cfg, optimizer_cdb)

    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)
    model_cdb, optimizer_cdb, = amp.initialize(model_cdb, optimizer_cdb, opt_level=amp_opt_level)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False, find_unused_parameters=True,
        )
        model_cdb = torch.nn.parallel.DistributedDataParallel(
            model_cdb, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False, find_unused_parameters=True,
        )

    arguments = {"iteration": 0, "iter_size": cfg.SOLVER.ITER_SIZE}
    output_dir = cfg.OUTPUT_DIR
    save_to_disk = get_rank() == 0
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    trainlog_period = cfg.SOLVER.TRAINLOG_PERIOD
    ckpt_output_dir = os.path.join(output_dir, 'ckpt')
    mkdir(ckpt_output_dir)
    # TODO: check whether the *_cdb is properly loaded for inference when using 1 GPU
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, ckpt_output_dir, save_to_disk, model_cdb=model_cdb
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT, model_only=not retrain)
    if load_optimizer:
        extra_checkpoint_data = checkpointer.load_optimizer(extra_checkpoint_data)
    if retrain:
        arguments["iteration"] = extra_checkpoint_data["iteration"]
        if cfg.SOLVER.STEPS != scheduler.milestones:
            print(f'Updating scheduler.milestones to {cfg.SOLVER.STEPS}')
            scheduler.milestones = cfg.SOLVER.STEPS

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )
    
    if use_tensorboard:
        meters = TensorboardLogger(
            log_dir=os.path.join(cfg['OUTPUT_DIR'], 'log/'),
            start_iter=arguments['iteration'],
            delimiter="  ")
    else:
        meters = MetricLogger(delimiter="  ")

    if cfg.SOLVER.ITER_SIZE == 1:
        do_train_cdb(
            model, model_cdb,
            data_loader,
            optimizer, optimizer_cdb,
            scheduler, scheduler_cdb,
            checkpointer,
            device,
            checkpoint_period,
            arguments,
            meters,
            trainlog_period,
            cfg
        )
    else:
        do_train_cdb_with_iter_size(
            model, model_cdb,
            data_loader,
            optimizer, optimizer_cdb,
            scheduler, scheduler_cdb,
            checkpointer,
            device,
            checkpoint_period,
            arguments,
            meters,
            trainlog_period,
            cfg
        )
    synchronize()
    return model


def run_test(cfg, model, distributed):
    if distributed:
        model = model.module
    iou_types = ("bbox",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()


def main():
    parser = argparse.ArgumentParser(description="wetectron training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--use-tensorboard",
        dest="use_tensorboard",
        help="Use tensorboardX logger (Requires tensorboardX installed)",
        action="store_true",
    )

    parser.add_argument(
        "--retrain",
        dest="retrain",
        help="Is this a continuation of another training",
        action="store_true",
    )

    parser.add_argument(
        "--load_optimizer",
        dest="load_optimizer",
        help="Load optimizer from existing checkpoint?",
        action="store_true",
    )   

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Check if model exists
    model_final_path = os.path.join(cfg.OUTPUT_DIR, 'ckpt', 'model_final.pth')
    if os.path.exists(model_final_path):
        print(f'ATTENTION: Model {model_final_path} already existing. \n\n')
        return
    else:
      mkdir(cfg.OUTPUT_DIR) # Creat output dir otherwise

    logger = setup_logger("wetectron", cfg.OUTPUT_DIR, get_rank(), 'train_logs.txt')

    # update parameters according to iter_size
    if not args.retrain:
        if cfg.DB.METHOD == "concrete":
            update_params_by_iter_size_and_npgus_cdb()
        else:
            update_params_by_iter_size_and_npgus()
    cfg.freeze()

    # make sure each worker has a different, yet deterministic seed if specified
    seed_all_rng(None if cfg.SEED < 0 else cfg.SEED + get_rank())

    logger.info(args)
    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())
    logger.info("Loaded configuration file {}".format(args.config_file))

    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)
    shutil.copyfile(args.config_file, Path(cfg.OUTPUT_DIR, 'original_config.yml'))
    
    if cfg.DB.METHOD == "concrete":
        model = train_cdb(
            cfg=cfg,
            local_rank=args.local_rank,
            distributed=args.distributed,
            use_tensorboard=args.use_tensorboard,
            retrain=args.retrain,
            load_optimizer = args.load_optimizer
        )
    else:
        model = train(
            cfg=cfg,
            local_rank=args.local_rank,
            distributed=args.distributed,
            use_tensorboard=args.use_tensorboard,
            retrain=args.retrain,
            load_optimizer = args.load_optimizer
        )
    
    if not args.skip_test:
        run_test(cfg, model, args.distributed)

def update_params_by_iter_size_and_npgus():
    # Update SOLVER.MAX_ITER to keep the number
    # of epoch unchanged.
    # Although the gradients are sum-aggregated for 'iter_size'
    # iterations before an update, the base learning rate is kept
    # unchanged w.r.t. 'iter_size' according to the linear scaling rule in 
    # Goyal et al., Training ImageNet in 1 Hour.
    # SOLVER.STEPS does not change because the scheduler will only
    # be stepped every 'iter_size' iterations.

    logger = logging.getLogger('wetectron')
    iter_size = cfg.SOLVER.ITER_SIZE
    logger.info(f"SOLVER.ITER_SIZE is set to {iter_size}.")

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    logger.info(f"Using {num_gpus} GPUs instead of 8.")
    old_base_lr = cfg.SOLVER.BASE_LR
    new_base_lr = old_base_lr / 8 * num_gpus
    cfg.SOLVER.BASE_LR = new_base_lr
    logger.info(f"SOLVER.BASE_LR: {old_base_lr} -> {new_base_lr}.")

    real_max_iter = cfg.SOLVER.MAX_ITER
    new_max_iter = int(real_max_iter * 8 / num_gpus)
    cfg.SOLVER.MAX_ITER = new_max_iter
    cfg.SOLVER.CHECKPOINT_PERIOD = int(cfg.SOLVER.CHECKPOINT_PERIOD * cfg.SOLVER.MAX_ITER / real_max_iter)
    logger.info(f"MAX_ITER: {real_max_iter} -> {new_max_iter}.")

def update_params_by_iter_size_and_npgus_cdb():
    # Update SOLVER.MAX_ITER to keep the number
    # of epoch unchanged.
    # Although the gradients are sum-aggregated for 'iter_size'
    # iterations before an update, the base learning rate is kept
    # unchanged w.r.t. 'iter_size' according to the linear scaling rule in 
    # Goyal et al., Training ImageNet in 1 Hour.
    # SOLVER.STEPS does not change because the scheduler will only
    # be stepped every 'iter_size' iterations.

    logger = logging.getLogger('wetectron')
    iter_size = cfg.SOLVER.ITER_SIZE
    logger.info(f"SOLVER.ITER_SIZE is set to {iter_size}.")
    
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    logger.info(f"Using {num_gpus} GPUs instead of 8.")
    
    old_base_lr = cfg.SOLVER.BASE_LR
    new_base_lr = old_base_lr / 8 * num_gpus
    cfg.SOLVER.BASE_LR = new_base_lr
    logger.info(f"SOLVER.BASE_LR: {old_base_lr} -> {new_base_lr}.")

    old_base_lr = cfg.SOLVER_CDB.BASE_LR
    new_base_lr = old_base_lr / 8 * num_gpus
    cfg.SOLVER_CDB.BASE_LR = new_base_lr
    logger.info(f"SOLVER_CDB.BASE_LR: {old_base_lr} -> {new_base_lr}.")

    real_max_iter = cfg.SOLVER.MAX_ITER
    new_max_iter = int(real_max_iter * 8 / num_gpus)
    cfg.SOLVER.MAX_ITER = new_max_iter
    cfg.SOLVER.CHECKPOINT_PERIOD = int(cfg.SOLVER.CHECKPOINT_PERIOD * 8 / num_gpus)
    logger.info(f"MAX_ITER: {real_max_iter} -> {new_max_iter}.")
    

if __name__ == "__main__":
    main()
