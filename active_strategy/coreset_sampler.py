#------------------------------------------------------------------------------
# Code developed by Huy V. Vo and Oriane Simeoni                              
# INRIA, Valeo.ai                                                             
#------------------------------------------------------------------------------

import argparse
import torch

import numpy as np
from tqdm import tqdm
from scipy.stats import entropy as ent

from active_sampler import BaseSampler
from loss_sampler import LossSampler

from scipy.spatial import distance_matrix
import torch.nn.functional as F

from wetectron.utils.comm import is_main_process

from pathlib import Path
from copy import deepcopy

def greedy_k_center(labeled, unlabeled, budget, STEP=100):
    # https://github.com/osimeoni/RethinkingDeepActiveLearning/blob/main/lib/selection_methods.py

    greedy_indices = []

    # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
    min_dist = np.min(distance_matrix(labeled[0, :].reshape((1, labeled.shape[1])), unlabeled),
                        axis=0)
    min_dist = min_dist.reshape((1, min_dist.shape[0]))
    for j in range(1, labeled.shape[0], STEP):
        if j + STEP < labeled.shape[0]:
            dist = distance_matrix(labeled[j:j+STEP, :], unlabeled)
        else:
            dist = distance_matrix(labeled[j:, :], unlabeled)
        min_dist = np.vstack((min_dist, np.min(dist, axis=0).reshape((1, min_dist.shape[1]))))
        min_dist = np.min(min_dist, axis=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))

    # iteratively insert the farthest index and recalculate the minimum distances:
    farthest = np.argmax(min_dist)
    greedy_indices.append(farthest)
    for i in tqdm(range(budget-1)):
        dist = distance_matrix(unlabeled[greedy_indices[-1], :].reshape((1,unlabeled.shape[1])),
                                unlabeled)
        min_dist = np.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
        min_dist = np.min(min_dist, axis=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))
        farthest = np.argmax(min_dist)
        greedy_indices.append(farthest)

    return np.array(greedy_indices)

def greedy_k_center_weigthed(labeled, unlabeled, u_weights, budget, STEP=100):
    # https://github.com/osimeoni/RethinkingDeepActiveLearning/blob/main/lib/selection_methods.py

    greedy_indices = []

    # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
    d = distance_matrix(labeled[0, :].reshape((1, labeled.shape[1])), unlabeled)
    # multiply distance by weight of unlabeled
    d = d*u_weights[np.newaxis, :]

    min_dist = np.min(d, axis=0)
    min_dist = min_dist.reshape((1, min_dist.shape[0]))
    for j in range(1, labeled.shape[0], STEP):
        if j + STEP < labeled.shape[0]:
            dist = distance_matrix(labeled[j:j+STEP, :], unlabeled)
        else:
            dist = distance_matrix(labeled[j:, :], unlabeled)
        # multiply distance by weight of unlabeled
        dist = dist*u_weights[np.newaxis, :]

        min_dist = np.vstack((min_dist, np.min(dist, axis=0).reshape((1, min_dist.shape[1]))))
        min_dist = np.min(min_dist, axis=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))

    # iteratively insert the farthest index and recalculate the minimum distances:
    farthest = np.argmax(min_dist)
    greedy_indices.append(farthest)
    for i in tqdm(range(budget-1)):
        dist = distance_matrix(unlabeled[greedy_indices[-1], :].reshape((1,unlabeled.shape[1])),
                                unlabeled)
        min_dist = np.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
        min_dist = np.min(min_dist, axis=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))
        farthest = np.argmax(min_dist)
        greedy_indices.append(farthest)

    return np.array(greedy_indices)

class CoreSetSampler(BaseSampler):
    """
    Select images using k-means over images.
    """

    def __init__(self, *args):
        super().__init__(*args)

        assert("_" in self.args.sel_method)

        # Get the nb of images to start coreset with
        self.init_nb = self.args.sel_method.split('_')[1]
        assert(self.init_nb.isdigit())
        self.init_nb = int(self.init_nb)

        self.weighted = None
        if 'wl-ent-max' in self.args.sel_method:
            self.weighted = 'wl-ent-max'
        elif 'wl-ent-sum' in self.args.sel_method:
            self.weighted = 'wl-ent-sum'
        elif 'wl-ent-mean' in self.args.sel_method:
            self.weighted = 'wl-ent-mean'
        elif 'wl' in self.args.sel_method:
            self.weighted = 'wl'
        

    def select(self): 
        print(f'Select images based on the clustering method: {self.args.sel_method}.')
        
        print('Loading model and data_loader ...')
        self.load_config()
        self.load_data()
        self.load_model()

        print('Load already selected images.')
        last_cycle_active_images, last_cycle_active_images_info = self.load_last_cycle_active_list()
        self.set_unlabelled_labelled_dataset(last_cycle_active_images)

        # Extract image features
        if self.args.cycle == 1: 
            assert(len(self.labelled_dataset) == 0)
            l_feats = []
            # Randomly select images to start with
            firsts = np.random.choice(len(self.unlabelled_dataset), self.init_nb)
        else:
            firsts = []
            if self.args.distributed:
                l_feats = self.extract_image_desc_distributed(dataset=self.labelled_dataset)
            else:
                l_feats = self.extract_image_desc(dataset=self.labelled_dataset)
        
        if self.args.distributed:
            u_feats = self.extract_image_desc_distributed(dataset=self.unlabelled_dataset)
        else:
            u_feats = self.extract_image_desc(dataset=self.unlabelled_dataset)

        if is_main_process():
            if self.args.cycle > 1:
                l_feats = np.vstack(l_feats)
            u_feats = np.vstack(u_feats)
        else:
            return None, None, None

        u_ind = range(len(self.unlabelled_dataset))
        if self.args.cycle ==1:
            u_ind = [i for i in u_ind if i not in firsts]

            l_feats = u_feats[np.array(firsts), :]
            u_feats = u_feats[np.array(u_ind), :]

        budget = self.args.budget if self.args.cycle > 1 else self.args.budget - self.init_nb

        # Apply greedy search with loss weights
        if self.weighted is None:
            print("Classic coreset.")
            selected = greedy_k_center(l_feats, u_feats, budget=budget)
        elif self.weighted == 'wl':
            print("Weight with loss")
            loss_args = deepcopy(self.args)
            loss_args.sel_method = self.args.bib_loss_method
            bib_loss_sampler = LossSampler(loss_args)
            loss_path = Path(
                    self.args.exp_name, "inference",
                    self.args.prediction_name, "losses.pth"
                )
            print(f'Loading losses from {str(loss_path)}')
            bib_loss_sampler.losses = torch.load(loss_path)
            loss_weights = bib_loss_sampler.pool_losses()

            u_weights = loss_weights[self.unlabelled_idxs[np.array(u_ind)]]

            print("Use weighted coreset.")
            selected = greedy_k_center_weigthed(l_feats, u_feats, u_weights=u_weights, budget=budget)

        elif "ent" in self.weighted:
            print("Weight with uncertainty")
             # Extract predictions
            prediction_path = Path(self.args.exp_name, 'inference',
                                self.args.prediction_name, 'predictions.pth')
            print(f'Loading predictions from {str(prediction_path)} ...')
            predictions = torch.load(prediction_path)
            assert len(predictions) == len(self.dataset), \
                f'Prediction length {len(predictions)} must be the same as dataset length {len(self.dataset)}!'

            # preds = [predictions[im_id] for im_id in]
            probs = [p.get_field("scores_all").cpu().numpy() for p in predictions]

            # Compute entropy per class, over all classes
            entropies = [ent(p, axis=1) for p in probs]

            # Aggregate box scores per image using max or mean 
            if "max" in self.weighted:
                print("Weight with max-uncertainty")
                values = [np.max(ent) for ent in entropies]
            elif "mean" in self.weighted:
                print("Weight with mean-uncertainty")
                values = [np.mean(ent) for ent in entropies]
            elif "sum" in self.weighted:
                print("Weight with sum-uncertainty")
                values = [np.sum(ent) for ent in entropies]

            u_weights = np.array(values)[self.unlabelled_idxs[np.array(u_ind)]]

            print("Use weighted coreset.")
            selected = greedy_k_center_weigthed(l_feats, u_feats, u_weights=u_weights, budget=budget)

        else:
            raise ValueError('Value of self.weighted not recognised!')

        # Get the list of image names
        u_indices = list(np.array(u_ind)[selected]) + list(firsts) # Add first indx to unlabelled
        image_indices = self.unlabelled_idxs[u_indices] # Got back to full data indexing
        active_images = [self.dataset.get_img_info(idx)['file_name'] for idx in image_indices]
        
        assert(len(np.unique(active_images)) == self.args.budget)

        # Sum to previous selection
        if int(self.args.cycle) > 1: # cycle in base 1
            image_indices = np.append(self.labelled_idxs, image_indices)
            active_images = self._previous_cycle_images + active_images

        save_info = {
                'init_nb': self.init_nb,
                'args': self.args
                }

        return active_images, image_indices, save_info
