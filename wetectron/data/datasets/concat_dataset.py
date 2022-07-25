#------------------------------------------------------------------------------
# Code adapted from https://github.com/NVlabs/wetectron
# by Huy V. Vo and Oriane Simeoni
# INRIA, Valeo.ai
#------------------------------------------------------------------------------

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import bisect

from torch.utils.data.dataset import ConcatDataset as _ConcatDataset
from wetectron.data import datasets

class ConcatDataset(_ConcatDataset):
    """
    Same as torch.utils.data.dataset.ConcatDataset, but exposes an extra
    method for querying the sizes of the image
    """

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self.get_idxs(idx)
        img, target, rois, _ = self.datasets[dataset_idx][sample_idx]
        return img, target, rois, idx

    def get_categories(self):
        return self.datasets[0].get_categories()

    def get_idxs(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx

    def get_img_info(self, idx):
        dataset_idx, sample_idx = self.get_idxs(idx)
        return self.datasets[dataset_idx].get_img_info(sample_idx)

    def get_active_images(self):
        return self.datasets[0].get_active_images()

    def is_active(self, idx):
        dataset_idx, sample_idx = self.get_idxs(idx)
        return self.datasets[dataset_idx].is_active(sample_idx)

    def get_active_sampling_weight(self, idx):
        dataset_idx, sample_idx = self.get_idxs(idx)
        return self.datasets[dataset_idx].get_active_sampling_weight(sample_idx)
        
    def get_weak_instance_weight(self, idx):
        dataset_idx, sample_idx = self.get_idxs(idx)
        return self.datasets[dataset_idx].get_weak_instance_weight(sample_idx)

    def has_pseudo_gt(self, idx):
        dataset_idx, sample_idx = self.get_idxs(idx)
        return self.datasets[dataset_idx].has_pseudo_gt(sample_idx)

    # Methods that only apply on a ConcatDataset of datasets.PascalVOCDataset
    def get_groundtruth(self, idx):
        dataset_idx, sample_idx = self.get_idxs(idx)
        return self.datasets[dataset_idx].get_groundtruth(sample_idx)

    def map_class_id_to_class_name(self, idx):
        dataset_idx, sample_idx = self.get_idxs(idx)
        return self.datasets[dataset_idx].map_class_id_to_class_name(sample_idx)