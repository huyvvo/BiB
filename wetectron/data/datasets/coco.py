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
import torch
import torchvision
import pickle5 as pickle
import numpy as np

from wetectron.structures.bounding_box import BoxList
from wetectron.structures.segmentation_mask import SegmentationMask
from wetectron.structures.keypoint import PersonKeypoints, Click
from wetectron.structures.boxlist_ops import remove_small_boxes, cat_boxlist

min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False

def unique_boxes(boxes, scale=1.0):
    """Return indices of unique boxes."""
    v = np.array([1, 1e3, 1e6, 1e9])
    hashes = np.round(boxes * scale).dot(v)
    _, index = np.unique(hashes, return_index=True)
    return np.sort(index)


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None, proposal_file=None,
        pseudo_boxes_file=None, active_images_file=None, weak_instance_weight_file=None
    ):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)
        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms
        self.ann_file = ann_file
        # Include proposals from a file
        if proposal_file is not None:
            print('Loading proposals from: {}'.format(proposal_file))
            with open(proposal_file, 'rb') as f:
                self.proposals = pickle.load(f, encoding='latin1')
            self.id_field = 'indexes' if 'indexes' in self.proposals else 'ids'  # compat fix
            # _sort_proposals(self.proposals, self.id_field)
            self.top_k = -1
        else:
            self.proposals = None

        if pseudo_boxes_file is not None:
            print('Loading pseudo boxes')
            with open(pseudo_boxes_file, 'rb') as f:
                self.pseudo_boxes = pickle.load(f)
        else:
            self.pseudo_boxes = None

        # active info
        if active_images_file is not None:
            with open(active_images_file, 'rb') as f:
                self.active_images = pickle.load(f)
            if isinstance(self.active_images, list):
                self.active_images = {k:1.0 for k in self.active_images}
            else:
                if not isinstance(self.active_images, dict):
                    raise Exception(f'active_images is of type {type(self.active_images)}'
                                    f'must be a list or a dictionary')
        else:
            self.active_images = None
        if self.active_images is not None:
            self.sum_active_weights = np.sum(list(self.active_images.values()))
        print(self.active_images)

        # weak instance weights
        if weak_instance_weight_file is not None:
            with open(weak_instance_weight_file, 'rb') as f:
                self.weak_instance_weights = pickle.load(f)
        else:
            self.weak_instance_weights = None

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        if "lvis_v0.5" not in self.ann_file:
            anno = [obj for obj in anno if obj["iscrowd"] == 0]

        if self.proposals is not None:
            img_id = self.ids[idx] 
            id_field = 'indexes' if 'indexes' in self.proposals else 'ids'  # compat fix
            roi_idx = self.proposals[id_field].index(img_id)
            rois = self.proposals['boxes'][roi_idx]

            # remove duplicate, clip, remove small boxes, and take top k
            keep = unique_boxes(rois)
            rois = rois[keep, :]
            # scores = scores[keep]
            rois = BoxList(torch.tensor(rois), img.size, mode="xyxy")
            rois = rois.clip_to_image(remove_empty=True)
            rois = remove_small_boxes(boxlist=rois, min_size=2)
            if self.top_k > 0:
                rois = rois[[range(self.top_k)]]
                # scores = scores[:self.top_k]
        else:
            rois = None

        # support un-labled
        if anno == [] and 'unlabeled' in self.ann_file:
            boxes = torch.as_tensor([[0,0,0,0]]).reshape(-1, 4)
            target = BoxList(boxes, img.size, mode="xyxy")
            classes = torch.tensor([0])
            target.add_field("labels", classes)

            if self.has_pseudo_gt(idx) and not self.is_active(idx):
                _p_boxes = self.pseudo_boxes[self.get_img_info(idx)['file_name']]
                _p_boxes = _p_boxes.resize(rois.size)
                rois = cat_boxlist((rois, BoxList(_p_boxes.bbox.clone(), rois.size, rois.mode)))
                rois.add_field('pseudo_boxes', _p_boxes)
                target.add_field('pseudo_boxes', _p_boxes)

            if self._transforms is not None:
                img, target, rois = self._transforms(img, target, rois)
            # add filename to target
            target.add_field('filename', self.get_img_info(idx)['file_name'])
            target.bbox.fill_(0)
        else:
            boxes = [obj["bbox"] for obj in anno]
            boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
            target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

            classes = [obj["category_id"] for obj in anno]
            classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
            classes = torch.tensor(classes)
            target.add_field("labels", classes)

            if anno and "segmentation" in anno[0]:
                masks = [obj["segmentation"] for obj in anno]
                masks = SegmentationMask(masks, img.size, mode='poly')
                target.add_field("masks", masks)

            if anno and "keypoints" in anno[0]:
                keypoints = [obj["keypoints"] for obj in anno]
                keypoints = PersonKeypoints(keypoints, img.size)
                target.add_field("keypoints", keypoints)

            if anno and 'point' in anno[0]:
                click = [obj["point"] for obj in anno]
                click = Click(click, img.size)
                target.add_field("click", click)

            if anno and 'scribble' in anno[0]:
                scribble = [obj["scribble"] for obj in anno]
                # xmin, ymin, xmax, ymax
                scribble_box = []
                for sc in scribble:
                    if len(sc[0]) == 0:
                        scribble_box.append([1, 2, 3, 4])
                    else:
                        scribble_box.append([min(sc[0]), min(sc[1]), max(sc[0]), max(sc[1])])
                scribble_box = torch.tensor(scribble_box)
                scribble_box = torch.as_tensor(scribble_box).reshape(-1, 4)  # guard against no boxes
                scribble_target = BoxList(scribble_box, img.size, mode="xyxy")
                target.add_field("scribble", scribble_target)

            if anno and 'use_as' in anno[0]:
                tag_to_ind = {'tag':0, 'point':1, 'scribble':2, 'box':3}
                use_as = [tag_to_ind[obj['use_as']] for obj in anno]
                use_as = torch.tensor(use_as)
                target.add_field("use_as", use_as)

            target = target.clip_to_image(remove_empty=True)

            if self.has_pseudo_gt(idx) and not self.is_active(idx):
                _p_boxes = self.pseudo_boxes[self.get_img_info(idx)['file_name']]
                _p_boxes = _p_boxes.resize(rois.size)
                rois = cat_boxlist((rois, BoxList(_p_boxes.bbox.clone(), rois.size, rois.mode)))
                rois.add_field('pseudo_boxes', _p_boxes)
                target.add_field('pseudo_boxes', _p_boxes)

            if self._transforms is not None:
                img, target, rois = self._transforms(img, target, rois)
            # add filename to target
            target.add_field('filename', self.get_img_info(idx)['file_name'])

        # add is_active Flag
        if self.is_active(idx):
            target.add_field('is_active', True)
            target.add_field(
                'active_instance_weight', 
                self.active_images[self.get_img_info(idx)['file_name']]/self.sum_active_weights*len(self.active_images)
            )
        else:
            target.add_field('is_active', False)
            if self.weak_instance_weights is not None and self.get_img_info(idx)['file_name'] in self.weak_instance_weights:
                target.add_field('weak_instance_weight', self.weak_instance_weights[self.get_img_info(idx)['file_name']])
            else:
                target.add_field('weak_instance_weight', 1.0)

        return img, target, rois, idx

    def get_groundtruth(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        if "lvis_v0.5" not in self.ann_file:
            anno = [obj for obj in anno if obj["iscrowd"] == 0]

        # support un-labled
        if anno == [] and 'unlabeled' in self.ann_file:
            boxes = torch.as_tensor([[0,0,0,0]]).reshape(-1, 4)
            target = BoxList(boxes, img.size, mode="xyxy")
            classes = torch.tensor([0])
            target.add_field("labels", classes)

            target.bbox.fill_(0)
        else:
            boxes = [obj["bbox"] for obj in anno]
            boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
            target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

            classes = [obj["category_id"] for obj in anno]
            classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
            classes = torch.tensor(classes)
            target.add_field("labels", classes)

            if anno and "segmentation" in anno[0]:
                masks = [obj["segmentation"] for obj in anno]
                masks = SegmentationMask(masks, img.size, mode='poly')
                target.add_field("masks", masks)

            if anno and "keypoints" in anno[0]:
                keypoints = [obj["keypoints"] for obj in anno]
                keypoints = PersonKeypoints(keypoints, img.size)
                target.add_field("keypoints", keypoints)

            if anno and 'point' in anno[0]:
                click = [obj["point"] for obj in anno]
                click = Click(click, img.size)
                target.add_field("click", click)

            if anno and 'scribble' in anno[0]:
                scribble = [obj["scribble"] for obj in anno]
                # xmin, ymin, xmax, ymax
                scribble_box = []
                for sc in scribble:
                    if len(sc[0]) == 0:
                        scribble_box.append([1, 2, 3, 4])
                    else:
                        scribble_box.append([min(sc[0]), min(sc[1]), max(sc[0]), max(sc[1])])
                scribble_box = torch.tensor(scribble_box)
                scribble_box = torch.as_tensor(scribble_box).reshape(-1, 4)  # guard against no boxes
                scribble_target = BoxList(scribble_box, img.size, mode="xyxy")
                target.add_field("scribble", scribble_target)

            if anno and 'use_as' in anno[0]:
                tag_to_ind = {'tag':0, 'point':1, 'scribble':2, 'box':3}
                use_as = [tag_to_ind[obj['use_as']] for obj in anno]
                use_as = torch.tensor(use_as)
                target.add_field("use_as", use_as)

            target = target.clip_to_image(remove_empty=True)

        return target

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data

    def get_active_images(self):
        return self.active_images

    # Does the image has full supervision?
    def is_active(self, index):
        if self.active_images is not None and self.get_img_info(index)['file_name'] in self.active_images:
            return True
        else:
            return False

    def get_active_sampling_weight(self, index):
        if self.active_images is not None and self.get_img_info(index)['file_name'] in self.active_images:
            return self.active_images[self.get_img_info(index)['file_name']]
        else:
            raise Exception('There are no active samples!')

    # Does the image have a pseudo box?
    def has_pseudo_gt(self, index):
        if self.pseudo_boxes is not None and self.get_img_info(index)['file_name'] in self.pseudo_boxes:
            return True
        else:
            return False

    def get_weak_instance_weight(self, index):
        if self.weak_instance_weights is not None and self.get_img_info(index)['file_name'] in self.weak_instance_weights:
            return self.weak_instance_weights[self.get_img_info(index)['file_name']]
        else:
            raise Exception('There is no weak instance weight!')

    def map_class_id_to_class_name(self, class_id):
        return self.categories[self.contiguous_category_id_to_json_id[class_id]]

    def get_categories(self):
        return {0:'__background__', **{cl:self.map_class_id_to_class_name(cl) for cl in self.contiguous_category_id_to_json_id}}

