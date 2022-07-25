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
from utils import kmeans_2plus_init_BiB, kmeans_2plus_init_BiB_with_cost
from pathlib import Path 
import numpy as np 
import torch
from tqdm import tqdm
import random
from sklearn.decomposition import PCA
from wetectron.utils.comm import is_main_process, synchronize

class Kmeans2PlusImageSampler(BaseSampler):
    """
    Apply kmeans++ initialization on image features.
    """

    def __init__(self, args):
        super(Kmeans2PlusImageSampler, self).__init__(args)    

    def build_cluster_data(self, rois_desc, rois_indices):
        """
        Extract box-in-box pair features and the info of each pair.

        Parameters:

            rois_desc (List of array): features of selected boxes in each image.
            rois_indices (dict): output from find_box_in_box_rois_from_predictions.

        Returns:

            X (List): each row of X contains features of a pair
            X_info (List): each row of X_info contains the information of a pair:
                index of the image, indices of the boxes in the original predictions,
                indices of the boxes in rois_desc, and their class. 
        """
        
        X = []
        X_info = []
        for idx in range(len(self.dataset)):
            if rois_indices[idx] is not None:
                assert(len(rois_desc[idx]) == len(rois_indices[idx]))
                X += [el.ravel() for el in rois_desc[idx]]
                X_info += [
                    [idx, contiguous_idx, real_idx] 
                    for contiguous_idx, real_idx in zip(range(len(rois_indices[idx])), 
                                                        rois_indices[idx])
                ]
        return X, X_info

    def _select_images(self, X, last_cycle_indices=None):
        if self.args.cycle == 1:
            first_image = random.choice(np.arange(len(X)))
            init_seeds = [first_image]
        else:
            init_seeds = list(last_cycle_indices)
        
        if self.args.unit == 'images':
            seed_indices, image_indices = kmeans_2plus_init_BiB.kmeans_plusplus(
                                            X, self.args.cycle * self.args.budget, 
                                            init_seeds, np.arange(len(X))
                                        )
            if len(image_indices) < self.args.cycle * self.args.budget:
                raise Exception('Cannot find enough centroids!')
            assert(set(image_indices) == set(seed_indices))

        elif self.args.unit == 'boxes':
            raise Exception('Not implemented!')
            
        return seed_indices, image_indices

    def select(self):
        print('Loading model and data_loader ...')
        self.load_config()
        self.load_data()
        self.load_model()
        
        last_cycle_active_images, last_cycle_active_images_info = self.load_last_cycle_active_list()
        self.set_unlabelled_labelled_dataset(last_cycle_active_images)

        #-----------
        print('Extracting image features')
        X = np.vstack(self.extract_image_desc(dataset=self.dataset))
        print('X_shape:', X.shape)

        #-----------
        
        print(f'Selecting images using {self.args.unit} as unit ...')
        if last_cycle_active_images_info is not None:
            last_cycle_active_images_indices = last_cycle_active_images_info['image_indices']
        else:
            last_cycle_active_images_indices = None
        seed_indices, image_indices = self._select_images(
                                            X, last_cycle_active_images_indices
                                        )
        # Get the list of image names
        active_images = [self.dataset.get_img_info(idx)['file_name'] for idx in image_indices]
        if last_cycle_active_images is not None:
            assert(set(last_cycle_active_images).issubset(set(active_images)))
        
        # Save info
        save_info = {
            'image_indices': image_indices,
            'args': self.args
        }

        return active_images, image_indices, save_info


class Kmeans2PlusSampler(BaseSampler):
    """
    Apply kmeans++ initialization on features of model's predictions.
    """

    def __init__(self, args):
        super(Kmeans2PlusSampler, self).__init__(args)
        
    def find_confident_predictions(self, predictions, score_thresh=0.5, return_indices=False):
        """
        Given the predictions (boxes) of all images and a score threshold, returns for each
        image the predictions whose confidence is greater than or equal to the threshold.
        
        Parameters:
        
            dataset:
            predictions: list of BoxList.
            score_thresh: float in [0,1].
            
        Returns:
        
        list 'preds', preds[i] contains the predictions of image i.
        list pred_indices, pred_indices[i] contains the indices of high-confidencet predictions in i.
        """

        assert(self._data_loaded)
        
        preds = [None] * len(self.dataset)
        pred_indices = [None] * len(self.dataset)
        for i in tqdm(range(len(self.dataset))):
            valid_ids = torch.nonzero(
                            predictions[i].get_field('scores') > score_thresh,
                            as_tuple=True
                        )[0]
            if len(valid_ids) > 0:
                valid_ids = valid_ids.tolist()
                pred_indices[i] = valid_ids
                preds[i] = predictions[i][valid_ids]
        if return_indices:
            return preds, pred_indices
        else:
            return preds    

    def build_cluster_data(self, rois_desc, rois_indices):
        """
        Extract box-in-box pair features and the info of each pair.

        Parameters:

            rois_desc (List of array): features of selected boxes in each image.
            rois_indices (dict): output from find_box_in_box_rois_from_predictions.

        Returns:

            X (List): each row of X contains features of a pair
            X_info (List): each row of X_info contains the information of a pair:
                index of the image, indices of the boxes in the original predictions,
                indices of the boxes in rois_desc, and their class. 
        """
        
        X = []
        X_info = []
        for idx in range(len(self.dataset)):
            if rois_indices[idx] is not None:
                assert(len(rois_desc[idx]) == len(rois_indices[idx]))
                X += [el.ravel() for el in rois_desc[idx]]
                X_info += [
                    [idx, contiguous_idx, real_idx] 
                    for contiguous_idx, real_idx in zip(range(len(rois_indices[idx])), 
                                                        rois_indices[idx])
                ]
        return X, X_info

    def _select_images_variant2(self, X, X_info, num_seeds_last_cycle):
        if self.args.cycle == 1:
            first_image = random.choice(np.unique(X_info[:,0]))
            init_seeds = list(np.where(X_info[:,0] == first_image)[0])
        else:
            init_seeds = list(range(num_seeds_last_cycle))
        
        if self.args.unit == 'images':
            seed_indices, image_indices = kmeans_2plus_init_BiB.kmeans_plusplus(
                                            X, self.args.cycle * self.args.budget, 
                                            init_seeds, X_info[:,0]
                                        )
            if len(image_indices) < self.args.cycle * self.args.budget:
                raise Exception('Cannot find enough centroids!')
            assert(set(image_indices) == set(np.unique(X_info[seed_indices,0])))

        elif self.args.unit == 'boxes':
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
        
        prediction_path = Path(self.args.exp_name, 'inference',
                               self.args.prediction_name, 'predictions.pth')
        print(f'Loading predictions from {str(prediction_path)} ...')
        predictions = torch.load(prediction_path)
        assert len(predictions) == len(self.dataset), \
            f'Prediction length {len(predictions)} must be the same as dataset length {len(self.dataset)}!'
        self.preds = predictions

        last_cycle_active_images, last_cycle_active_images_info = self.load_last_cycle_active_list()
        self.set_unlabelled_labelled_dataset(last_cycle_active_images)

        #-----------
        print('Finding box-in-box pairs ...')
        rois, rois_indices = self.find_confident_predictions(
                                    predictions, self.args.score_thresh, return_indices=True
                                )
        if last_cycle_active_images_info is not None:
            # remove box pairs in images that were selected in the last cycle
            for idx in last_cycle_active_images_info['centroids_info'][:,0]:
                rois_indices[idx] = None
                rois[idx] = last_cycle_active_images_info['rois_per_image'][idx]
        if self.args.distributed:
            synchronize()
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
        
        X, X_info = self.build_cluster_data(rois_desc, rois_indices)
        if last_cycle_active_images_info is not None:
            last_X, last_X_info = [], list(last_cycle_active_images_info['centroids_info'])
            for _i in range(len(last_X_info)):
                idx = last_X_info[_i][0]
                last_X.append(rois_desc[idx][last_X_info[_i][1]])
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
        if self.args.bib_variant == 2:
            seed_indices, image_indices = self._select_images_variant2(
                                            X, X_info, len(last_X_info)
                                        )
        else:
            raise ValueError(f'Value of args.variant ({self.args.variant}) not supported!')

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
