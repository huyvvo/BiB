#------------------------------------------------------------------------------
# Code adapted from https://github.com/NVlabs/wetectron
# by Huy V. Vo and Oriane Simeoni
# INRIA, Valeo.ai
#------------------------------------------------------------------------------

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .coco import COCODataset
from .voc import PascalVOCDataset
from .concat_dataset import ConcatDataset

def _isinstance(dataset, dataset_type):
    if isinstance(dataset, ConcatDataset):
        membership = [isinstance(dataset.datasets[i], dataset_type) for i in range(len(dataset.datasets))]
        assert(membership.count(membership[0]) == len(membership))
        return membership[0]
    else:
        return isinstance(dataset, dataset_type)

__all__ = ["COCODataset", "ConcatDataset", "PascalVOCDataset", "_isinstance"]
