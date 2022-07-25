#------------------------------------------------------------------------------
# Code developed by Huy V. Vo and Oriane Simeoni                              
# INRIA, Valeo.ai                                                             
#------------------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
from wetectron.utils.env import setup_environment  # noqa F401 isort:skip
import wetectron
print(wetectron.__file__)

import os
import numpy as np
import pickle5 as pickle
import torch
from torch.utils.data import Subset
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt

from wetectron.config import cfg
from wetectron.data import make_data_loader
from wetectron.modeling.detector import build_detection_model
from wetectron.utils.checkpoint import DetectronCheckpointer
from wetectron.data import transforms as T
from wetectron.structures.image_list import to_image_list
from wetectron.utils.visualize import overlay_boxes
from wetectron.modeling.cdb import ConvConcreteDB
from wetectron.utils.comm import get_world_size, is_main_process, synchronize

import logging
import torch.nn.functional as F

class BaseSampler(object):
    def __init__(self, args):
        self.args = args
        self._data_loaded = False
        self._model_loaded = False
        self._config_loaded = False

        # Initialization
        self.preds = None

    def get_active_list_path(self, cycle=None, nb_selected=None):
        if cycle is None:
            cycle = self.args.cycle
        if nb_selected is None:
            nb_selected = self.args.cycle * self.args.budget

        save_dir = Path(f'{self.args.output_dir}')
        if self.args.unit == 'images':
            save_name = Path(save_dir, f'cycle{cycle}_{nb_selected}_images.pkl')
        elif self.args.unit == 'boxes':
            save_name = Path(save_dir, f'cycle{cycle}_{nb_selected}_boxes.pkl')
        else:
            raise ValueError(f'unit must be in ("image", "box"), not {self.args.unit}')
        return save_dir, save_name

    def save_active_list(self, active_images, save_info=None):
        save_dir, save_name = self.get_active_list_path()
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f'Saving selected images to {save_name}')
        if Path(save_name).exists():
            raise Exception(f'File {save_name} exists! Using another value of the --ver argument to save result!')
        with open(save_name, 'wb') as f:
            pickle.dump(active_images, f, protocol=pickle.HIGHEST_PROTOCOL)
        if save_info is not None:
            with open(str(save_name).replace('.pkl', '_info.pkl'), 'wb') as f:
                pickle.dump(save_info, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_last_cycle_active_list(self):
        if self.args.cycle > 1:
            print('Loading last cycle selection ...')
            load_dir, load_name = self.get_active_list_path(
                                        int(self.args.cycle)-1,
                                        self.args.budget * (self.args.cycle-1)
                                    )
            with open(load_name, 'rb') as f:
                last_cycle_active_images = pickle.load(f)
            with open(str(load_name).replace('.pkl', '_info.pkl'), 'rb') as f:
                last_cycle_active_images_info = pickle.load(f)

            print(f'There are {len(last_cycle_active_images)} images already selected.')
            return last_cycle_active_images, last_cycle_active_images_info
        else:
            print('First cycle ...')
            return None, None

    def load_config(self):
        # TODO: Adapt the code for other dataset
        self.args.opts = ['OUTPUT_DIR', self.args.exp_name,
                 'MODEL.WEIGHT', f'{self.args.exp_name}/ckpt/model_final.pth',
                 'TEST.CONCAT_DATASETS', True, 
                 'TEST.BBOX_AUG.ENABLED', True, # make this parameter true to perform augmentation later
                 'TEST.REMOVE_IMAGES_WITHOUT_ANNOTATIONS', True, 
                 'TEST.IMS_PER_BATCH', 8
                 ]
        cfg.merge_from_file(self.args.config_file)
        cfg.merge_from_list(self.args.opts)
        # Update to run inference on train
        cfg.DATASETS.TEST = cfg.DATASETS.TRAIN
        cfg.PROPOSAL_FILES.TEST = cfg.PROPOSAL_FILES.TRAIN
        print(f'cfg.DATASETS.TEST: {cfg.DATASETS.TEST}')
        print(f'cfg.PROPOSAL_FILES.TEST: {cfg.PROPOSAL_FILES.TEST}')
        if self.args.use_cdb:
            cfg.DB.METHOD = "concrete"
        cfg.freeze()
        self.cfg = cfg

        self._config_loaded = True

    def load_data(self):
        """ Load the dataset 
        """
        data_loaders_val = make_data_loader(self.cfg, is_train=False, is_distributed=False)
        self.data_loader = data_loaders_val[0]
        self.dataset = self.data_loader.dataset

        # See if data is transformed in data_loader
        print('Collate function:', self.data_loader.collate_fn)
        self.transform = T.Compose([
            T.Resize(self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MAX_SIZE_TEST),
            T.ToTensor(),
            T.Normalize(
                mean=self.cfg.INPUT.PIXEL_MEAN, 
                std=self.cfg.INPUT.PIXEL_STD, 
                to_bgr255=self.cfg.INPUT.TO_BGR255
            )
        ])
        self._data_loaded = True

    def set_unlabelled_labelled_dataset(self, selected):
        """ Generate labelled/unlabelled datasets.
        """
        if selected == None:
            self.labelled_dataset = []
            self.labelled_idxs = np.array([])
            self.unlabelled_dataset = self.dataset
            self.unlabelled_idxs = np.array(range(len(self.dataset)))

        else:
            l = [idx for idx in range(len(self.dataset)) if self.dataset.get_img_info(idx)['file_name'] in selected]
            u = [idx for idx in range(len(self.dataset)) if self.dataset.get_img_info(idx)['file_name'] not in selected]

            self.labelled_dataset = Subset(self.dataset, l)
            self.labelled_idxs = np.array(l)
            print(f'len labelled dataset: {len(self.labelled_dataset)} imgs')

            self.unlabelled_dataset = Subset(self.dataset, u)
            self.unlabelled_idxs = np.array(u)
            print(f'len unlabelled dataset: {len(self.unlabelled_dataset)} imgs')

        if len(self.unlabelled_dataset) + len(self.labelled_dataset) != len(self.dataset):
            raise ValueError(f'len(U) [{len(self.unlabelled_dataset)}] + len(L) [{len(self.labelled_dataset)}] != len(D) [{len(self.dataset)}]')

        self._previous_cycle_images = selected
        self._data_loaded = True

    def load_model(self):
        model = build_detection_model(self.cfg)
        model.to(self.cfg.MODEL.DEVICE)
        self.device = self.cfg.MODEL.DEVICE
        
        # Adversarial model
        model_cdb = None
        if cfg.DB.METHOD == "concrete":
            model_cdb = ConvConcreteDB(self.cfg, model.backbone.out_channels)
            model_cdb.to(self.cfg.MODEL.DEVICE)    
        ckpt_dir = os.path.join(self.cfg.OUTPUT_DIR, 'ckpt')
        checkpointer = DetectronCheckpointer(self.cfg, model, save_dir=ckpt_dir, model_cdb=model_cdb)
        _ = checkpointer.load(self.cfg.MODEL.WEIGHT, use_latest=self.cfg.MODEL.WEIGHT is None)

        model.eval()
        if self.cfg.DB.METHOD == "concrete":
            model_cdb.eval()
        
        # Store the model used next
        self.model = model
        self.model_cdb = model_cdb

        self._model_loaded = True

    def extract_image_desc_iter(self, images, device='cuda'):
        """
        Perform image feature extraction.
        """    
        t_images = []
        for image in images:
            t_img, _, _ = self.transform(image)
            t_images.append(t_img)
        t_images = to_image_list(t_images, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        
        with torch.no_grad():
            features = self.model.backbone(t_images.to(device).tensors)
        return features

    def extract_image_desc(self, dataset=None, device='cuda'):
        """
        Extract image features.
        """
        timer = None # Timer()
        if dataset is None:
            dataset = self.dataset
        desc = [None] * len(dataset)
        for i in tqdm(range(len(dataset))):
            image, target, _, image_id = dataset[i]
            d = self.extract_image_desc_iter([image], device)[0]
            # Apply global avg pooling
            d = F.avg_pool2d(d, (d.shape[2], d.shape[3])).squeeze().cpu().numpy()
            desc[i] = d
        return desc

    def extract_image_desc_distributed_aux(self, dataset=None, device='cuda'):
        """
        rois: external boxes
        """
        num_gpus = get_world_size()
        indices = list(range(len(dataset)))
        if len(indices) % num_gpus > 0:
            indices += [len(indices)-1] * (num_gpus - len(indices) % num_gpus)
        indices = np.array(indices)
        indices = np.split(indices, num_gpus)

        desc = {}
        if is_main_process():
            iters = tqdm(indices[self.args.local_rank])
        else:
            iters = indices[self.args.local_rank]
        for i in iters:
            image, target, _, image_id = self.dataset[i]
            im_feat = self.extract_image_desc_iter([image], device)[0]

            im_feat = F.avg_pool2d(im_feat, (im_feat.shape[2], im_feat.shape[3])).squeeze().cpu().numpy()
            desc.update({i:im_feat})

        save_dir, save_name = self.get_active_list_path()
        desc_dir = Path(save_dir, f'image_desc_cycle{self.args.cycle}')
        desc_dir.mkdir(parents=True, exist_ok=True)
        with open(Path(desc_dir, f'gpu{self.args.local_rank}.pkl'), 'wb') as f:
            pickle.dump(desc, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _accumulate_image_desc_from_multiple_gpus(self):
        num_gpus = get_world_size()
        save_dir, _ = self.get_active_list_path()
        all_image_desc = []
        for _gpu in range(num_gpus):
            print(f'Reading image desc from image_desc_cycle{self.args.cycle}/gpu{_gpu}.pkl')
            desc_path = Path(save_dir, f'image_desc_cycle{self.args.cycle}/gpu{_gpu}.pkl')
            with open(desc_path, 'rb') as f:
                all_image_desc.append(pickle.load(f))

        image_desc = {}
        for p in all_image_desc:
            image_desc.update(p)
        # convert a dict where the key is the index in a list
        image_ids = list(sorted(image_desc.keys()))
        if len(image_ids) != image_ids[-1] + 1:
            logger = logging.getLogger("wetectron.inference")
            logger.warning(
                "Number of images that were gathered from multiple processes is not "
                "a contiguous set. Some images might be missing from the computation"
            )

        # convert to a list
        image_desc = [image_desc[i] for i in image_ids]
        return image_desc

    def extract_image_desc_distributed(self, dataset=None, device='cuda'):
        """
        rois: external boxes
        """
        self.extract_image_desc_distributed_aux(dataset, device='cuda')
        synchronize()
        if not is_main_process():
            return
        image_desc = self._accumulate_image_desc_from_multiple_gpus()
        return image_desc

    def extract_roi_features_iter(self, images, targets, device, rois, model_cdb=None):
        """
        Perform roi feature extraction.
        """    
        t_images = []
        t_rois = []
        for image, roi in zip(images, rois):  
            t_img, _, t_roi = self.transform(image, rois=roi)
            t_images.append(t_img)
            t_rois.append(t_roi)
        t_images = to_image_list(t_images, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        t_rois = [r.to(device) if r is not None else None for r in t_rois]

        with torch.no_grad():
            features = self.model.backbone(t_images.to(device).tensors)
            x, result, detector_losses, accuracy = self.model.roi_heads(features, t_rois, targets, model_cdb)
        return x.split([len(roi) for roi in rois])

    def extract_roi_features(self, rois, device='cuda', model_cdb=None):
        """
        rois: external boxes
        """
        timer = None # Timer()
        rois_desc = [None] * len(self.dataset)
        for i in tqdm(range(len(self.dataset))):
            if rois[i] is not None:
                image, target, _, image_id = self.dataset[i]
                rois_desc[i] = self.extract_roi_features_iter(
                                    [image], [target], device, [rois[i]], model_cdb
                                )[0].to(torch.device('cpu')).numpy()
        return rois_desc

    def extract_roi_features_distributed_aux(self, rois, device='cuda', model_cdb=None):
        """
        rois: external boxes
        """
        num_gpus = get_world_size()
        indices = list(range(len(self.dataset)))
        if len(indices) % num_gpus > 0:
            indices += [len(indices)-1] * (num_gpus - len(indices) % num_gpus)
        indices = np.array(indices)
        indices = np.split(indices, num_gpus)

        rois_desc = {}
        if is_main_process():
            iters = tqdm(indices[self.args.local_rank])
        else:
            iters = indices[self.args.local_rank]
        for i in iters:
            if rois[i] is not None:
                image, target, _, image_id = self.dataset[i]
                _rois_desc = self.extract_roi_features_iter(
                                    [image], [target], device, [rois[i]], model_cdb
                                )
                rois_desc.update({i:_rois_desc[0].to(torch.device('cpu')).numpy()})
            else:
                rois_desc.update({i:None})
        save_dir, save_name = self.get_active_list_path()
        desc_dir = Path(save_dir, f'desc_cycle{self.args.cycle}')
        desc_dir.mkdir(parents=True, exist_ok=True)
        with open(Path(desc_dir, f'gpu{self.args.local_rank}.pkl'), 'wb') as f:
            pickle.dump(rois_desc, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _accumulate_rois_desc_from_multiple_gpus(self):
        num_gpus = get_world_size()
        save_dir, _ = self.get_active_list_path()
        all_rois_desc = []
        for _gpu in range(num_gpus):
            print(f'Reading rois desc from '
                  f'desc_cycle{self.args.cycle}/gpu{_gpu}.pkl'
            )
            desc_path = Path(
                save_dir, 
                f'desc_cycle{self.args.cycle}/gpu{_gpu}.pkl'
            )
            with open(desc_path, 'rb') as f:
                all_rois_desc.append(pickle.load(f))

        rois_desc = {}
        for p in all_rois_desc:
            rois_desc.update(p)
        # convert a dict where the key is the index in a list
        image_ids = list(sorted(rois_desc.keys()))
        if len(image_ids) != image_ids[-1] + 1:
            logger = logging.getLogger("wetectron.inference")
            logger.warning(
                "Number of images that were gathered from multiple processes is not "
                "a contiguous set. Some images might be missing from the computation"
            )

        # convert to a list
        rois_desc = [rois_desc[i] for i in image_ids]
        return rois_desc

    def extract_roi_features_distributed(self, rois, device='cuda', model_cdb=None):
        """
        rois: external boxes
        """
        self.extract_roi_features_distributed_aux(rois, device='cuda', model_cdb=model_cdb)
        synchronize()
        if not is_main_process():
            return
        rois_desc = self._accumulate_rois_desc_from_multiple_gpus()
        return rois_desc

    def select(self, nb_imgs):
        """ Not implemented.
        """
        return None

    def visualize_active_images(self, selected_indices, predictions=None):
        # TODO: adapt the code for other datasets
        vis_path = Path(f'{self.args.vis_path}/{self.args.save_name_prefix}_ver{self.args.ver}')
        Path(vis_path).mkdir(parents=True, exist_ok=True)
        print(f'Selected images are visualized in {self.args.vis_path} ...')
        for idx in selected_indices:
            img = self.dataset[idx][0]
            img_cv = cv2.imread('datasets/voc/VOC2007/' + self.dataset.get_img_info(idx)['file_name'])
            if predictions is not None and predictions[idx] is not None:
                preds = predictions[idx].to(torch.device('cpu'))
                scores = preds.get_field('scores')
                img_cv = overlay_boxes(img_cv, preds[scores >= 0.5].resize(img.size))
            plt.figure()
            plt.imsave(Path(vis_path, f'{idx}.jpg'),cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    def get_class_histogram(self, active_images):
        """
        get_class_histogram(dataset, active_images)

        Compute the number of images of each class in 'active_images'

        Parameter:

            dataset:
            active_images: str, pathlib.Path, list, array of dictionary.
                str or pathlib.Path: path to an active list
                array, list: contains image names
                dict: keys are image names, values are image weights
        Return:

            dict of (cl,freq) where cl is a class index and freq is the number of images
                of that class in 'active_images'.
        """
        assert(self._data_loaded)
        if isinstance(active_images, str) or isinstance(active_images, Path):
            with open(active_images, 'rb') as f: 
                active_images = pickle.load(f)
        if isinstance(active_images, np.ndarray) or isinstance(active_images, list):
            active_images = {el:1 for el in active_images}
        else:
            if not isinstance(active_images, dict):
                raise Exception(f'active_images of type {type(active_images)} not supported!')

        active_ids = [im_id for im_id in range(len(self.dataset)) 
                      if self.dataset.get_img_info(im_id)['file_name'] in active_images]
        num_classes = len(list(self.dataset.get_categories().keys()))
        class_histogram = {cl:0 for cl in range(num_classes)}
        for im_id in active_ids:
            file_name = self.dataset.get_img_info(im_id)['file_name']
            labels = self.dataset.get_groundtruth(im_id).get_field('labels').unique()
            for j in labels:
                class_histogram[j.item()] += active_images[file_name]
        return class_histogram

