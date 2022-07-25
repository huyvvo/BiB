#------------------------------------------------------------------------------
# Code adapted from https://github.com/NVlabs/wetectron
# by Huy V. Vo and Oriane Simeoni
# INRIA, Valeo.ai
#------------------------------------------------------------------------------

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import bisect
import copy
import logging
import numpy as np
import pdb
from tqdm import tqdm
from pathlib import Path

import torch.utils.data
from wetectron.utils.comm import get_world_size
from wetectron.utils.imports import import_file
from wetectron.utils.miscellaneous import save_labels, seed_all_rng
from wetectron.utils.model_zoo import cache_url

from . import datasets as D
from . import samplers

from .collate_batch import BatchCollator, BBoxAugCollator
from .transforms import build_transforms


def build_dataset(dataset_list, transforms, dataset_catalog, 
                  is_train=True, use_difficult=None, proposal_files=None, 
                  pseudo_boxes_file=None, active_images_file=None, 
                  weak_instance_weight_file=None, concat_dataset=False,
                  remove_images_without_annotations=False):
    """
    Arguments:
        dataset_list (list[str]): Contains the names of the datasets, i.e.,
            coco_2014_train, coco_2014_val, etc
        transforms (callable): transforms to apply to each (image, target) sample
        dataset_catalog (DatasetCatalog): contains the information on how to
            construct a dataset.
        is_train (bool): whether to setup the dataset for training or testing
    """
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )
    if proposal_files is None or len(proposal_files) == 0:
        proposal_files = (None, ) * len(dataset_list)
    assert len(dataset_list) == len(proposal_files)
        
    datasets = []
    for index, dataset_name in enumerate(dataset_list):
        is_labeled = "unlabeled" not in dataset_name
        data = dataset_catalog.get(dataset_name)
        factory = getattr(D, data["factory"])
        args = data["args"]
        # for COCODataset, we want to remove images without annotations during training
        if data["factory"] == "COCODataset":
            args["remove_images_without_annotations"] = (is_train and is_labeled) or remove_images_without_annotations
        if data["factory"] == "PascalVOCDataset":
            if use_difficult is None:
                args["use_difficult"] = not is_train
            else:
                args["use_difficult"] = use_difficult
        args["transforms"] = transforms
        
        # load proposal
        _f = proposal_files[index]
        if _f is not None and _f.startswith("http"):
            # if the file is a url path, download it and cache it
            _f = cache_url(_f)            
        args["proposal_file"] = _f

        args["pseudo_boxes_file"] = pseudo_boxes_file
        args["active_images_file"] = active_images_file
        args["weak_instance_weight_file"] = weak_instance_weight_file

        # make dataset from factory
        dataset = factory(**args)
        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train and not concat_dataset:
        return datasets

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = D.ConcatDataset(datasets)

    return [dataset]


def make_data_sampler(dataset, shuffle, distributed, weights=None):
    if distributed:
        if weights is not None:
            print("WeightedDistributedSampler")
            return samplers.WeightedDistributedSampler(dataset, weights, shuffle=shuffle)
        else:
            return samplers.DistributedSampler(dataset, shuffle=shuffle)
    elif weights is not None:
        print("WeightedRandomSampler")
        return torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        img_info = dataset.get_img_info(i)
        aspect_ratio = float(img_info["height"]) / float(img_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_data_sampler(
    dataset, sampler, aspect_grouping, images_per_batch, num_iters=None, start_iter=0
):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, images_per_batch, drop_uneven=False
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=False
        )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler

def make_data_loader(cfg, is_train=True, is_distributed=False, start_iter=0):
    num_gpus = get_world_size()
    if is_train:
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = True
        num_iters = cfg.SOLVER.MAX_ITER
    else:
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = False if not is_distributed else True
        num_iters = None
        start_iter = 0

    if images_per_gpu > 1:
        logger = logging.getLogger(__name__)
        logger.warning(
            "When using more than one image per GPU you may encounter "
            "an out-of-memory (OOM) error if your GPU does not have "
            "sufficient memory. If this happens, you can reduce "
            "SOLVER.IMS_PER_BATCH (for training) or "
            "TEST.IMS_PER_BATCH (for inference). For training, you must "
            "also adjust the learning rate and schedule length according "
            "to the linear scaling rule. See for example: "
            "https://github.com/facebookresearch/Detectron/blob/master/configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml#L14"
        )

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []

    paths_catalog = import_file(
        "wetectron.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
    proposal_files = cfg.PROPOSAL_FILES.TRAIN if is_train else cfg.PROPOSAL_FILES.TEST
    pseudo_boxes_file = cfg.ACTIVE.PSEUDO_BOXES_FILE if is_train else None

    # If bbox aug is enabled in testing, simply set transforms to None and we will apply transforms later
    transforms = None if not is_train and cfg.TEST.BBOX_AUG.ENABLED else build_transforms(cfg, is_train)
    datasets = build_dataset(dataset_list, transforms, DatasetCatalog, 
                             is_train, None, proposal_files, pseudo_boxes_file,
                             cfg.ACTIVE.INPUT_FILE, cfg.WEAK_INSTANCE_WEIGHT_FILE,
                             cfg.TEST.CONCAT_DATASETS, cfg.TEST.REMOVE_IMAGES_WITHOUT_ANNOTATIONS)
    if is_train:
        # save category_id to label name mapping
        save_labels(datasets, cfg.OUTPUT_DIR)

    data_loaders = []
    for dataset in datasets:
        
        weights, aspect_grouping = build_sampling_weights(cfg, dataset, is_train, aspect_grouping)
        sampler = make_data_sampler(dataset, shuffle, is_distributed, weights=weights)
        batch_sampler = make_batch_data_sampler(
            dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter
        )
        collator = BBoxAugCollator() if not is_train and cfg.TEST.BBOX_AUG.ENABLED else \
            BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
        num_workers = cfg.DATALOADER.NUM_WORKERS
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            worker_init_fn=worker_init_reset_seed
        )
        data_loaders.append(data_loader)
    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders

def build_sampling_weights(cfg, dataset, is_train, aspect_grouping):
    """
    
    Compute a sampling weight for each image in the training set.
    
    If ACTIVE.WEIGHTED_SAMPLING is on, fully-annotated images are set to 
    the same weight, and weakly-annotated are also set to the same weight.
    The weights of weakly- and fully-annotated images are chosen such that
    the ratio of fully-annotated images among those sampled by the Sampler
    is asymtotically ACTIVE.ACTIVE_SAMPLE_RATIO.

    """

    # IMPORTANT parameter, the sampler will not use the sampling weights
    # properly if aspect_grouping is set to True
    aspect_grouping = False
        
    weights = None
    weights_active = None
    # cannot use both weak and active weighting scheme simultaneously
    assert not (cfg.WEAK_WEIGHTED_SAMPLING and cfg.ACTIVE.WEIGHTED_SAMPLING)
    
    if is_train and cfg.ACTIVE.WEIGHTED_SAMPLING:
        print('Using active weighted sampling ...')
        # get indices of active samples
        active_index = [im_id for im_id in range(len(dataset)) if dataset.is_active(im_id)]
        inactive_index = list(set(range(len(dataset))) - set(active_index))
        
        if cfg.ACTIVE.LOSS.USE_INSTANCE_WEIGHT:
            # if using instance weights for loss, do not use instance sampling weight
            # All samples have equal sampling weights
            active_sampling_weights = np.array([1/len(active_index)]*len(active_index))
        else:
            active_sampling_weights = [dataset.get_active_sampling_weight(im_id) for im_id in active_index]
            active_sampling_weights = np.array(active_sampling_weights)/np.sum(active_sampling_weights)

        if cfg.ACTIVE.BATCH_ONLY_FULLY_SUP:    
            inactive_sampling_weights = np.zeros(len(inactive_index))
        else:
            weak_data_ratio = (1-cfg.ACTIVE.ACTIVE_SAMPLE_RATIO)/cfg.ACTIVE.ACTIVE_SAMPLE_RATIO
            inactive_sampling_weights = weak_data_ratio * np.ones(len(inactive_index)) / len(inactive_index)

        weights_active = np.zeros(len(dataset))
        weights_active[active_index] = active_sampling_weights
        weights_active[inactive_index] = inactive_sampling_weights
        
        if weights_active is not None:
            weights = weights_active

        if cfg.DEBUG.LOG_ACTIVE_SAMPLES_INFO:
            print('Active sampling weights:\n', active_sampling_weights)
            print('Active indices:\n', active_index)

        torch.save({dataset.get_img_info(im_id)['file_name'] : weights[im_id] for im_id in range(len(dataset))}, 
                   Path(cfg.OUTPUT_DIR, 'active_instance_weight.pth'))


    if is_train and cfg.WEAK_WEIGHTED_SAMPLING:
        print('Using weak weighted sampling ...')
        weak_weights = [dataset.get_weak_instance_weight(im_id) for im_id in range(len(dataset))]
        weights = np.array(weak_weights)/np.sum(weak_weights)
        torch.save({dataset.get_img_info(im_id)['file_name'] : weights[im_id] for im_id in range(len(dataset))}, 
                   Path(cfg.OUTPUT_DIR, 'weak_instance_weight.pth'))

    return weights, aspect_grouping

def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2 ** 31) + worker_id)