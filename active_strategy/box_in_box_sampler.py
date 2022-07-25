#------------------------------------------------------------------------------
# Code developed by Huy V. Vo and Oriane Simeoni                              
# INRIA, Valeo.ai                                                             
#------------------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
from wetectron.utils.env import setup_environment  # noqa F401 isort:skip
from wetectron.structures.boxlist_ops import boxlist_iou_async

from active_sampler import BaseSampler
from loss_sampler import LossSampler
from utils import kmeans_2plus_init_BiB, kmeans_2plus_init_BiB_with_cost
from pathlib import Path 
import numpy as np 
import torch
from tqdm import tqdm
import collections
import random
from copy import deepcopy
from sklearn.decomposition import PCA
from wetectron.utils.comm import is_main_process, synchronize

class BoxInBoxSampler(BaseSampler):
    def __init__(self, args):
        super(BoxInBoxSampler, self).__init__(args)
        self.min_cover = 0.8
        self.min_area_ratio = args.bib_min_area_ratio

    def find_confident_predictions(self, predictions, class_idx, score_thresh=0.5, 
                                   show_progress=True, return_indices=False):
        """
        Given the predictions (boxes) of all images, a class and a score threshold, returns for each
        image the predictions corresponding to the class whose confidence is greater than or equal 
        to the threshold.
        
        Parameters:
        
            dataset:
            predictions: list of BoxList.
            class_idx: int, class index.
            score_thresh: float in [0,1].
            
        Returns:
        
        dictionaray 'preds', preds[i] contains the predictions of image i.
        """
        assert(self._data_loaded)
        if show_progress: 
            image_ids = tqdm(range(len(self.dataset)))
        else: 
            image_ids = range(len(self.dataset))
        
        preds, pred_indices = {}, {}
        for i in image_ids:
            gt = self.dataset.get_groundtruth(i)
            if class_idx in gt.get_field('labels'):
                valid_ids = torch.nonzero(torch.logical_and(
                                                predictions[i].get_field('labels') == class_idx,
                                                predictions[i].get_field('scores') > score_thresh),
                                         as_tuple=True)[0]
                if len(valid_ids) > 0:
                    valid_ids = valid_ids.tolist()
                    pred_indices[i] = valid_ids
                    preds[i] = predictions[i][valid_ids]
        if return_indices:
            return preds, pred_indices
        else:
            return preds

    def find_confident_predictions_all_classes(self, predictions, score_thresh=0.5, 
                                               return_indices=False):
        """
        Given the predictions (boxes) of all images, a class and a score threshold, returns for each
        image the predictions corresponding to the class whose confidence is greater than or equal 
        to the threshold.
        
        Parameters:
        
            dataset:
            predictions: list of BoxList.
            score_thresh: float in [0,1].
            
        Returns:
        
        dictionaray 'preds', preds[i] contains the predictions of image i.
        """
        assert(self._data_loaded)
        num_classes = len(self.dataset.get_categories())

        preds = {cl:{} for cl in range(1, num_classes)}
        pred_indices = {cl:{} for cl in range(1, num_classes)}

        for i in tqdm(range(len(self.dataset))):
            gt = self.dataset.get_groundtruth(i)
            for class_idx in gt.get_field('labels').unique():
                valid_ids = torch.nonzero(torch.logical_and(
                                                predictions[i].get_field('labels') == class_idx,
                                                predictions[i].get_field('scores') > score_thresh),
                                         as_tuple=True)[0]
                if len(valid_ids) > 0:
                    valid_ids = valid_ids.tolist()
                    pred_indices[class_idx.item()][i] = valid_ids
                    preds[class_idx.item()][i] = predictions[i][valid_ids]
        if return_indices:
            return preds, pred_indices
        else:
            return preds    

    def box_in_box_same_class(self, boxlist):
        """
        Find box-in-box pairs. (A,B) is a pair of (small box, big box) if at least 
        'min_cover' of A's area is in B and B's area is at least 'min_area_ratio' of A's.

        Parameters:

            boxlist: wetectron.structures.bounding_box.BoxList object
            min_cover: float
            min_area_ratio: float

        Returns:

            list of tuples (i,j) where i and j are indices of the boxes in a box-in-box pairs.
        """
        assert torch.min(boxlist.get_field('labels') == boxlist.get_field('labels')[0]) > 0
        bbox_area = (boxlist.bbox[:,2]-boxlist.bbox[:,0])*(boxlist.bbox[:,3]-boxlist.bbox[:,1])
        bbox_covered_perc = boxlist_iou_async(boxlist, boxlist)
        
        box_in_box_pairs = []
        for i in range(len(boxlist)):
            bbox_covered_perc[i,i] = -1
            ids = torch.nonzero(torch.logical_and(
                                    bbox_covered_perc[i] >= self.min_cover, 
                                    bbox_area >= self.min_area_ratio * bbox_area[i]
                                ), as_tuple=False)
            if len(ids) > 0: 
                box_in_box_pairs += [[i, j] for j in ids[0].tolist()]
            
        return box_in_box_pairs

    def find_box_in_box_rois_from_predictions(self, predictions):
        """
        Parameters:

            args: Namespace, arguments of the experiment.
            predictions (List of BoxList): predictions of a detector.

        Returns:

            rois (List of BoxList): rois[idx] contains the boxes in box-in-box pairs of image 'idx'. 
                rois[idx] is None if image 'idx' has no pair.
            rois_indices (dict): rois_indices[idx] contains the indices of the boxes in box-in-box
                pairs of image 'idx': rois[idx] = predictions[idx][rois_indices[idx]]  
            box_pair_per_class (dict of dict): box_pair_per_class[cl][idx] contains pairs of box indices
                of box-in-box pairs of class 'cl' in image 'idx'.

        """
        assert(self._data_loaded)
        num_classes = len(self.dataset.get_categories())
        box_pair_per_class = {cl:{} for cl in range(1, num_classes)}

        all_preds, all_pred_indices = self.find_confident_predictions_all_classes(
                                        predictions, 
                                        score_thresh=self.args.score_thresh, 
                                        return_indices=True
                                    )
        for cl in tqdm(range(1,num_classes)):
            preds, pred_indices = all_preds[cl], all_pred_indices[cl]
            box_pair_per_class[cl] = {}
            for idx in preds:
                box_pairs = self.box_in_box_same_class(preds[idx])
                c_pred_indices = pred_indices[idx]
                if len(box_pairs) > 0:
                    box_pair_per_class[cl][idx] = [[c_pred_indices[bp[0]],c_pred_indices[bp[1]]] for bp in box_pairs]
                        
        # Gather rois for each image, rois might correspond to different classes
        # rois_indices[idx] contains the indices of the boxes in image 'idx' that
        # are in a pair.
        rois_indices = {}
        for cl in range(1, num_classes):
            for idx in box_pair_per_class[cl]:
                if idx in rois_indices:
                    rois_indices[idx] += list(np.concatenate(box_pair_per_class[cl][idx]))
                else:
                    rois_indices[idx] = list(np.concatenate(box_pair_per_class[cl][idx]))
        for idx in rois_indices:
            rois_indices[idx] = np.unique(rois_indices[idx])

        rois = [None] * len(self.dataset)
        for idx in rois_indices:
            rois[idx] = predictions[idx][rois_indices[idx]]

        return rois, rois_indices, box_pair_per_class

    def build_cluster_data(self, rois_desc, rois_indices, box_pair_per_class):
        """
        Extract box-in-box pair features and the info of each pair.

        Parameters:

            rois_desc (List of array): features of selected boxes in each image.
            rois_indices (dict): output from find_box_in_box_rois_from_predictions.
            box_pair_per_class (dict): output from find_box_in_box_rois_from_predictions.

        Returns:

            X (List): each row of X contains features of a pair
            X_info (List): each row of X_info contains the information of a pair:
                index of the image, indices of the boxes in the original predictions,
                indices of the boxes in rois_desc, and their class. 
        """
        
        X = [] 
        X_info = []
        num_classes = len(self.dataset.get_categories())
        for cl in range(1,num_classes):
            for idx, pairs in box_pair_per_class[cl].items():
                for pi in pairs:
                    small_idx = np.where(rois_indices[idx] == pi[0])[0][0]
                    large_idx = np.where(rois_indices[idx] == pi[1])[0][0]
                    feat = rois_desc[idx][(small_idx, large_idx),:].ravel()
                    X.append(feat)
                    X_info.append([idx, pi[0], pi[1], small_idx, large_idx, cl])
        return X, X_info

    def _select_images(self, X, X_info, num_seeds_last_cycle, total_sel=None):
        if self.args.cycle == 1:
            image_freq = collections.Counter(X_info[:,0]).most_common()
            freq = [el[1] for el in image_freq]
            pos = np.searchsorted(-np.array(freq), -freq[0], 'right')
            first_image = random.choice([el[0] for el in image_freq[:pos]])
            init_seeds = list(np.where(X_info[:,0] == first_image)[0])
        else:
            init_seeds = list(range(num_seeds_last_cycle))
        
        if total_sel is None:
            total_sel = self.args.cycle * self.args.budget
 
        if self.args.unit == 'images':
            seed_indices, image_indices = kmeans_2plus_init_BiB.kmeans_plusplus(
                                            X, total_sel,
                                            init_seeds, X_info[:,0]
                                        )
            if len(image_indices) < total_sel:
                raise Exception('Cannot find enough centroids!')
            assert(set(image_indices) == set(np.unique(X_info[seed_indices,0])))

        elif self.args.unit == 'boxes':
            if total_sel is not None:
                raise ValueError("Should not happen.")
            group_cost = {k:len(self.dataset.get_groundtruth(k)) 
                              for k in np.unique(X_info[:,0])}
            max_cost = self.args.budget * self.args.cycle
            seed_indices, image_indices = kmeans_2plus_init_BiB_with_cost.kmeans_plusplus(
                                            X, init_seeds, X_info[:,0],
                                            group_cost, max_cost
                                        )
            assert len(image_indices) == len(np.unique(image_indices))
            sum_cost = np.sum([len(self.dataset.get_groundtruth(im_id)) 
                               for im_id in image_indices])
            print(f"Max_cost: {max_cost}, actual_cost: {sum_cost}")
            print(f'Last image contains '
                  f'{len(self.dataset.get_groundtruth(image_indices[-1]))} objects'
            )
            print(f'{len(image_indices)} images selected!')
            
        return seed_indices, image_indices

    def select(self):
        print('Loading model and data_loader ...')
        self.load_config()
        self.load_data()
        self.load_model()
        num_classes = len(self.dataset.get_categories())
        
        prediction_path = Path(self.args.exp_name, 'inference',
                               self.args.prediction_name, 'predictions.pth')
        print(f'Loading predictions from {str(prediction_path)} ...')
        predictions = torch.load(prediction_path)
        assert len(predictions) == len(self.dataset), \
            f'Prediction length {len(predictions)} must be the same as dataset length {len(self.dataset)}!'
        self.preds = predictions

        last_cycle_active_images, last_cycle_active_images_info = self.load_last_cycle_active_list()
        self.set_unlabelled_labelled_dataset(last_cycle_active_images)

        last_random = [] # Used for variant3
        #-----------
        print('Finding box-in-box pairs ...')
        rois, rois_indices, box_pair_per_class = self.find_box_in_box_rois_from_predictions(predictions)
        if last_cycle_active_images_info is not None:
            # remove box pairs in images that were selected in the last cycle
            for idx in last_cycle_active_images_info['centroids_info'][:,0]:
                rois_indices.pop(idx, None)
                for cl in range(1, num_classes):
                    box_pair_per_class[cl].pop(idx, None)
                if self.args.bib_variant == 3:
                    assert idx not in last_random
                if not self.args.bib_variant == 2:
                    rois[idx] = last_cycle_active_images_info['rois_per_image'][idx]

            if self.args.bib_variant == 3:
                last_random = last_cycle_active_images_info['randomly_selected']

                for idx in last_random:
                    # Make sure to remove all randomly selected images from ROIS
                    rois_indices.pop(idx, None)
                    for cl in range(1, num_classes):
                        box_pair_per_class[cl].pop(idx, None)
                    rois[idx] = None
        if self.args.distributed:
            synchronize()

        if self.args.bib_variant == 2:
            print("Randomly select between images with pairs.")
            possibles = list(rois_indices.keys())
            selected = random.sample(possibles, self.args.budget)

            if last_cycle_active_images_info is not None:
                selected.extend(list(last_cycle_active_images_info['centroids_info'][:,0]))

            # Get the list of image names
            active_images = [self.dataset.get_img_info(idx)['file_name'] for idx in selected]
            if last_cycle_active_images is not None:
                assert(set(last_cycle_active_images).issubset(set(active_images)))

            # Save info
            save_info = {
                'centroids_info': np.array(selected)[:,None],
                'args': self.args
            }

            return active_images, selected, save_info

        print('Extracting rois features ...')
        if self.args.distributed:
            rois_desc = self.extract_roi_features_distributed(
                            rois, device='cuda', model_cdb=self.model_cdb
                        )
            if not is_main_process():
                return None, None, None
        else:
            rois_desc = self.extract_roi_features(
                            rois, device='cuda', model_cdb=self.model_cdb
                        )
        print(f'Length rois_desc: {len(rois_desc)}')
        print('Total number of rois features:', np.sum([el.shape[0] for el in rois_desc if el is not None]))
        
        #-----------
        
        X, X_info = self.build_cluster_data(rois_desc, rois_indices, box_pair_per_class)
        if last_cycle_active_images_info is not None:
            last_X, last_X_info = [], list(last_cycle_active_images_info['centroids_info'])
            for _i in range(len(last_X_info)):
                idx = last_X_info[_i][0]
                last_X.append(rois_desc[idx][last_X_info[_i][3:5]].ravel())
            X = last_X + X
            X_info = last_X_info + X_info
        else:
            last_X, last_X_info = [], []

        X, X_info = np.array(X), np.array(X_info)
        print('X_shape:', X.shape)
        if self.args.bib_pca_dim > 0:
            print(f'Performing PCA to reduce feature dimension from {X.shape[1]} to {self.args.bib_pca_dim}')
            X = PCA(self.args.bib_pca_dim).fit_transform(X)
        #-----------
        
        print(f'Selecting images using {self.args.unit} as unit ...')
        if self.args.bib_variant == 1:
            seed_indices, image_indices = self._select_images(X, X_info, len(last_X_info))
        elif self.args.bib_variant == 3:
            print("Half random + BiB pairs.")
            half_budget = int(self.args.budget/2)
            seed_indices, image_indices = self._select_images(X, X_info, len(last_X_info), 
                                                              total_sel = self.args.cycle*half_budget)

            bib_image_indices = image_indices.copy()

            # Add BiB to random selected
            already_used = list(image_indices)

            if last_cycle_active_images_info is not None:
                already_used += last_random

            # Randomly select some
            possible_indices = [im for im in range(len(self.dataset)) if im not in already_used]
            randomly_selected = random.sample(possible_indices, half_budget)

            # Add newly selected to previous randomly selected ones
            randomly_selected.extend(last_random)

            # Increase
            try:
                image_indices = list(image_indices)
                image_indices.extend(randomly_selected)
            except:
                breakpoint()

            # Get the list of image names
            active_images = [self.dataset.get_img_info(idx)['file_name'] for idx in image_indices]

            # Save info
            save_info = {
                'centroids_info': X_info[seed_indices],
                'rois_per_image': {im_id:rois[im_id] for im_id in bib_image_indices},
                'randomly_selected': randomly_selected,
                'args': self.args
            }

            return active_images, image_indices, save_info
        else:
            raise ValueError(f'Value of args.bib_variant ({self.args.bib_variant}) not supported!')

        # Get the list of image names
        active_images = [self.dataset.get_img_info(idx)['file_name'] for idx in image_indices]
        if last_cycle_active_images is not None:
            assert(set(last_cycle_active_images).issubset(set(active_images)))
        
        # Save info
        save_info = {
            'centroids_info': X_info[seed_indices],
            'rois_per_image': {im_id:rois[im_id] for im_id in image_indices},
            'args': self.args
        }

        return active_images, image_indices, save_info
