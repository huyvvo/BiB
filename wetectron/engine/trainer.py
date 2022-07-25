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
import datetime
import logging
import time
import torch
from apex import amp
import pdb
from pathlib import Path
from wetectron.structures.bounding_box import BoxList
from wetectron.utils.comm import get_world_size, get_rank, synchronize


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        torch.distributed.reduce(all_losses, dst=0)
        if torch.distributed.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

def update_momentum(optimizer, cur_lr, new_lr, logger, SCALE_MOMENTUM_THRESHOLD = 1.1, eps = 1e-10):
    """Update momentum as Sutskever et. al. and implementations in some other frameworks."""
    import numpy as np
    ratio = np.max((new_lr / np.max((cur_lr, eps)), cur_lr / np.max((new_lr, eps))))
    if ratio > SCALE_MOMENTUM_THRESHOLD:
        logger.info("update_momentum")
        param_keys = []
        for ind, param_group in enumerate(optimizer.param_groups):
            param_keys += param_group['params']
        correction = new_lr / cur_lr
        for p_key in param_keys:
            param_state = optimizer.state[p_key]
            if 'momentum_buffer' in param_state:
                param_state['momentum_buffer'] *= correction    

def do_train(
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
):
    logger = logging.getLogger("wetectron.trainer")
    logger.info("Start training")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    iter_size = arguments["iter_size"]
    model.train()
    start_training_time = time.time()
    end = time.time()

    for iteration, (images, targets, rois, _) in enumerate(data_loader, start_iter):
        meters.increase_counter()
        if any(len(target) < 1 for target in targets):
            logger.error(f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
            continue
        # add active information
        assert(len(targets) == 1)
        if cfg.ACTIVE.WEIGHTED_SAMPLING and cfg.ACTIVE.BATCH_ONLY_FULLY_SUP:
            if targets[0].get_field('is_active'):
                class_histogram = {f'class_{cl}':torch.tensor(cl in targets[0].get_field('labels')).float().to(device) 
                                   for cl in range(len(data_loader.dataset.get_categories()))}
        else:
            class_histogram = {f'class_{cl}':torch.tensor(cl in targets[0].get_field('labels')).float().to(device) 
                               for cl in range(len(data_loader.dataset.get_categories()))}
        
        if targets[0].get_field('is_active') == True:
            targets[0].add_field('active', torch.tensor([True]*len(targets[0])))
            rois = (rois[0].copy_with_fields([]),)
            if cfg.DEBUG.LOG_ACTIVE_IMAGE_NAME:
                logger.info(f"Active image {targets[0].get_field('filename')}")
        data_time = time.time() - end
        
        # Only update learning rate every 'iter_size' iterations
        if iteration % iter_size == 0:
            cur_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            new_lr = optimizer.param_groups[0]["lr"]
            if cur_lr > 1e-7 and cur_lr != new_lr:
                update_momentum(optimizer, cur_lr, new_lr, logger)
        meters.update(lr=optimizer.param_groups[0]["lr"])
        iteration = iteration + 1
        arguments["iteration"] = iteration
    
        images = images.to(device)
        targets = [target.to(device) for target in targets]
        rois = [r.to(device) if r is not None else None for r in rois]

        loss_dict, metrics = model(images, targets, rois)
        losses = sum(loss for loss in loss_dict.values())
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)
        # accuracy
        metrics_reduced = reduce_loss_dict(metrics)
        meters.update(**metrics_reduced)

        # class frequency
        if cfg.ACTIVE.WEIGHTED_SAMPLING and cfg.ACTIVE.BATCH_ONLY_FULLY_SUP:
            if targets[0].get_field('is_active'):     
                freq_reduced = reduce_loss_dict(class_histogram)
                meters.update(**freq_reduced)
        else:
            freq_reduced = reduce_loss_dict(class_histogram)
            meters.update(**freq_reduced)

        
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()

        # Only update parameters' gradient every 'iter_size' iterations
        if iteration % iter_size == 0:
            optimizer.step()
            optimizer.zero_grad()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        # Logging
        trainlog_period = trainlog_period if trainlog_period != -1 else 20*iter_size

        gpu_rank = get_rank()
        if iteration % trainlog_period == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "rank: {gpu_rank}",
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.8f}",
                        "max mem: {memory:.0f}\n",
                    ]
                ).format(
                    gpu_rank=gpu_rank,
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )


def do_train_cdb(
    model,
    model_cdb,
    data_loader,
    optimizer,
    optimizer_cdb,
    scheduler,
    scheduler_cdb,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    meters,
    trainlog_period,
    cfg
):
    logger = logging.getLogger("wetectron.trainer")
    logger.info("Start training")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    iter_size = arguments["iter_size"]
    model.train()
    model_cdb.train()
    start_training_time = time.time()
    end = time.time()
    
    for iteration, (images, targets, rois, _) in enumerate(data_loader, start_iter):
        meters.increase_counter()
        if any(len(target) < 1 for target in targets):
            logger.error(f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
            continue
        # add active information
        assert(len(targets) == 1)
        if cfg.ACTIVE.WEIGHTED_SAMPLING and cfg.ACTIVE.BATCH_ONLY_FULLY_SUP:
            if targets[0].get_field('is_active'):
                class_histogram = {f'class_{cl}':torch.tensor(cl in targets[0].get_field('labels')).float().to(device) 
                                   for cl in range(len(data_loader.dataset.get_categories()))}
        else:
            class_histogram = {f'class_{cl}':torch.tensor(cl in targets[0].get_field('labels')).float().to(device) 
                               for cl in range(len(data_loader.dataset.get_categories()))}
        
        if targets[0].get_field('is_active') == True:
            targets[0].add_field('active', torch.tensor([True]*len(targets[0])))
            rois = (rois[0].copy_with_fields([]),)
            if cfg.DEBUG.LOG_ACTIVE_IMAGE_NAME:
                logger.info(f"Active image {targets[0].get_field('filename')}")

        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        scheduler_cdb.step()
        new_lr = optimizer.param_groups[0]["lr"]
        if cur_lr > 1e-7 and cur_lr != new_lr:
            update_momentum(optimizer, cur_lr, new_lr, logger)
            update_momentum(optimizer_cdb, cur_lr, new_lr, logger)
        
        images = images.to(device)
        targets = [target.to(device) for target in targets]
        rois = [r.to(device) if r is not None else None for r in rois]

        loss_dict, metrics = model(images, targets, rois, model_cdb)
        losses = sum(loss for loss in loss_dict.values())
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)
        
        # accuracy
        metrics_reduced = reduce_loss_dict(metrics)
        meters.update(**metrics_reduced)
        optimizer.zero_grad()
        optimizer_cdb.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()
        optimizer.step()

        # concrete db
        optimizer.zero_grad()
        optimizer_cdb.zero_grad()
        losses_cdb = torch.tensor(0.0)
        if not targets[0].get_field('is_active') or cfg.ACTIVE.GO_THROUGH_CDB:
            loss_dict, metrics = model(images, targets, rois, model_cdb)
            losses_cdb = - float(cfg.DB.WEIGHT) * sum(loss for loss in loss_dict.values())
            with amp.scale_loss(losses_cdb, optimizer_cdb) as scaled_losses_cdb:
                scaled_losses_cdb.backward()
            optimizer_cdb.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        # Logging
        trainlog_period = trainlog_period if trainlog_period != -1 else 20*iter_size

        gpu_rank = get_rank()
        if iteration % trainlog_period == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [   
                        "rank: {gpu_rank}",
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.8f}",
                        "lr_cdb: {lr_cdb:.8f}",
                        "max mem: {memory:.0f}",
                        'loss_cdb: {loss_cdb:.4f}\n',
                    ]
                ).format(
                    gpu_rank=gpu_rank,
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    lr_cdb=optimizer_cdb.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    loss_cdb=losses_cdb.item(),
                )
            )

        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

def do_train_cdb_with_iter_size(
    model,
    model_cdb,
    data_loader,
    optimizer,
    optimizer_cdb,
    scheduler,
    scheduler_cdb,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    meters,
    trainlog_period,
    cfg
):
    logger = logging.getLogger("wetectron.trainer")
    logger.info("Start training")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    iter_size = arguments["iter_size"]
    model.train()
    model_cdb.train()
    start_training_time = time.time()
    end = time.time()

    trainlog_period = trainlog_period if trainlog_period != -1 else 20*iter_size
    assert trainlog_period % iter_size == 0, 'trainlog_period must be divisible by iter_size' 
    assert checkpoint_period % iter_size == 0, 'checkpoint_period must be divisible by iter_size'
    assert max_iter % iter_size == 0, 'max_iter must be divisible by iter_size'
            

    images_list = []
    targets_list = []
    rois_list = []
    
    for iteration, (images, targets, rois, _) in enumerate(data_loader, start_iter):
        meters.increase_counter()
        if any(len(target) < 1 for target in targets):
            logger.error(f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
            continue
        # add active information
        assert(len(targets) == 1)
        if cfg.ACTIVE.WEIGHTED_SAMPLING and cfg.ACTIVE.BATCH_ONLY_FULLY_SUP:
            if targets[0].get_field('is_active'):
                class_histogram = {f'class_{cl}':torch.tensor(cl in targets[0].get_field('labels')).float().to(device) 
                                   for cl in range(len(data_loader.dataset.get_categories()))}
        else:
            class_histogram = {f'class_{cl}':torch.tensor(cl in targets[0].get_field('labels')).float().to(device) 
                               for cl in range(len(data_loader.dataset.get_categories()))}
        
        if targets[0].get_field('is_active') == True:
            targets[0].add_field('active', torch.tensor([True]*len(targets[0])))
            rois = (rois[0].copy_with_fields([]),)
            if cfg.DEBUG.LOG_ACTIVE_IMAGE_NAME:
                logger.info(f"Active image {targets[0].get_field('filename')}")

        images = images.to(device)
        targets = [target.to(device) for target in targets]
        rois = [r.to(device) if r is not None else None for r in rois]

        images_list.append(images)
        targets_list.append(targets)
        rois_list.append(rois)

        if len(images_list) == iter_size:
            data_time = time.time() - end
            cur_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            scheduler_cdb.step()
            new_lr = optimizer.param_groups[0]["lr"]
            if cur_lr > 1e-7 and cur_lr != new_lr:
                update_momentum(optimizer, cur_lr, new_lr, logger)
                update_momentum(optimizer_cdb, cur_lr, new_lr, logger)

            # normal gradient step
            optimizer.zero_grad()
            optimizer_cdb.zero_grad()
            for images, targets, rois in zip(images_list, targets_list, rois_list):
                meters.increase_counter()
                loss_dict, metrics = model(images, targets, rois, model_cdb)
                losses = sum(loss for loss in loss_dict.values())
                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = reduce_loss_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                meters.update(loss=losses_reduced, **loss_dict_reduced)
                # accuracy
                metrics_reduced = reduce_loss_dict(metrics)
                meters.update(**metrics_reduced)
                # Note: If mixed precision is not used, this ends up doing nothing
                # Otherwise apply loss scaling for mixed-precision recipe
                with amp.scale_loss(losses, optimizer) as scaled_losses:
                    scaled_losses.backward()
            optimizer.step()
            
            for _ in range(iter_size): 
                meters.decrease_counter()

            # adversarial gradient step
            optimizer.zero_grad()
            optimizer_cdb.zero_grad()
            mean_losses_cdb = 0
            num_pass_cdb = 0
            for images, targets, rois in zip(images_list, targets_list, rois_list):
                if not targets[0].get_field('is_active') or cfg.ACTIVE.GO_THROUGH_CDB:
                    loss_dict, metrics = model(images, targets, rois, model_cdb)
                    losses_cdb = - float(cfg.DB.WEIGHT) * sum(loss for loss in loss_dict.values())
                    mean_losses_cdb = (mean_losses_cdb * num_pass_cdb + losses_cdb.item())/(num_pass_cdb+1)
                    num_pass_cdb += 1
                    with amp.scale_loss(losses_cdb, optimizer_cdb) as scaled_losses_cdb:
                        scaled_losses_cdb.backward()
            # concrete db
            optimizer_cdb.step()

            # clear data cache
            images_list = []
            targets_list = []
            rois_list = []

            batch_time = time.time() - end
            end = time.time()
            for _ in range(iter_size):
                meters.increase_counter()
                meters.update(time=batch_time/iter_size, data=data_time/iter_size)
            
            iteration = iteration + 1
            arguments["iteration"] = iteration
            
            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            # Logging
            gpu_rank = get_rank()
            if iteration % trainlog_period == 0 or iteration == max_iter:
                logger.info(
                    meters.delimiter.join(
                        [   
                            "rank: {gpu_rank}",
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.8f}",
                            "lr_cdb: {lr_cdb:.8f}",
                            "max mem: {memory:.0f}",
                            'loss_cdb: {loss_cdb:.4f}\n',
                        ]
                    ).format(
                        gpu_rank=gpu_rank,
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        lr_cdb=optimizer_cdb.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                        loss_cdb=mean_losses_cdb,
                    )
                )

            if iteration % checkpoint_period == 0:
                checkpointer.save("model_{:07d}".format(iteration), **arguments)
            if iteration == max_iter:
                checkpointer.save("model_final", **arguments)
            synchronize()

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
