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
import errno
import json
import logging
import os
import shutil
import socket
import datetime
from .comm import is_main_process
from datetime import datetime
import random
import numpy as np
import torch
from pathlib import Path

def get_run_name():
    """ A unique name for each run """
    return datetime.now().strftime(
        '%b%d_%H_%M_%S') + '_' + socket.gethostname()

def copy_source_code(output_dir):
    os.makedirs(output_dir)
    p = Path(__file__).parents[2]
    checklist = ['apex', 'build', 'configs', 'setup.py', 
                 'tools', 'wetectron', 'wetectron.egg-info']
    except_list = ['LICENSE', 'outputs', 'README.md',
                   'datasets', 'docs', 'proposal', 'notebooks']
    Fs = os.listdir(p)
    assert(set(checklist).issubset(set(Fs)))
    to_copy = [el for el in Fs if el not in except_list]
    # return 
    for f in to_copy:
        if Path(p,f).is_dir():
            shutil.copytree(Path(p,f), Path(output_dir,f), symlinks=True)
        else:
            shutil.copyfile(Path(p,f), Path(output_dir,f), follow_symlinks=False)

def seed_all_rng(seed=None):
    """
    Set the random seed for the RNG in torch, numpy and python.
    Args:
        seed (int): if None, will use a strong random seed.
    """
    if seed is None:
        seed = (
            os.getpid()
            + int(datetime.now().strftime("%S%f"))
            + int.from_bytes(os.urandom(2), "big")
        )
        logger = logging.getLogger(__name__)
        logger.info("Using a generated random seed {}".format(seed))
    np.random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    random.seed(seed)

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def save_labels(dataset_list, output_dir):
    if is_main_process():
        logger = logging.getLogger(__name__)

        ids_to_labels = {}
        for dataset in dataset_list:
            if hasattr(dataset, 'categories'):
                ids_to_labels.update(dataset.categories)
            else:
                logger.warning("Dataset [{}] has no categories attribute, labels.json file won't be created".format(
                    dataset.__class__.__name__))

        if ids_to_labels:
            labels_file = os.path.join(output_dir, 'labels.json')
            logger.info("Saving labels mapping into {}".format(labels_file))
            with open(labels_file, 'w') as f:
                json.dump(ids_to_labels, f, indent=2)


def save_config(cfg, path):
    if is_main_process():
        with open(path, 'w') as f:
            f.write(cfg.dump())
