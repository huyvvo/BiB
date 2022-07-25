#------------------------------------------------------------------------------
# Code developed by Huy V. Vo and Oriane Simeoni                              
# INRIA, Valeo.ai                                                             
#------------------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
from wetectron.utils.env import setup_environment  # noqa F401 isort:skip

from active_sampler import BaseSampler
from pathlib import Path 
import numpy as np 


class RandomSampler(BaseSampler):
    def __init__(self, args):
        super(RandomSampler, self).__init__(args)
        self.args.save_name_prefix = f'{self.args.save_name_prefix}_variant_{self.args.random_variant}'

    def get_inactive_id_by_class(self, last_cycle_active_images):
        """
        get_inactive_id_by_class(self, last_cycle_active_images)
        
        Return:

            dict of (cl, list) where cl is a class index and 'list' contains the index of
                inactive images of class 'cl'.
        """
        num_classes = len(list(self.dataset.get_categories().keys()))
        inactive_id_by_class = {cl:[] for cl in range(1, num_classes)}
        for im_id in range(len(self.dataset)):
            if self.dataset.get_img_info(im_id)['file_name'] not in last_cycle_active_images:
                for j in self.dataset.get_groundtruth(im_id).get_field('labels').unique():
                    inactive_id_by_class[j.item()].append(im_id)
        return inactive_id_by_class
    
    def _select_uniform(self, last_cycle_active_images):
        """
        Select the images uniformly randomly.
        """
        active_indices = [im_id for im_id in range(len(self.dataset)) 
                          if self.dataset.get_img_info(im_id)['file_name'] in last_cycle_active_images]
        choices = list(set(range(len(self.dataset))) - set(active_indices))
        num_images_to_select = self.args.budget
        if self.args.unit == "images":
            new_ids = list(np.random.choice(choices, num_images_to_select, replace=False))
            active_indices = active_indices + new_ids
        elif self.args.unit == "boxes":
            if len(active_indices) > 0:
                current_cost = np.sum([len(self.dataset.get_groundtruth(im_id)) 
                                       for im_id in active_indices])
            else:
                current_cost = 0
            max_cost = self.args.budget * self.args.cycle
            choices = np.random.permutation(choices)
            for c in choices:
                if current_cost >= max_cost:
                    break
                active_indices.append(c)
                current_cost += len(self.dataset.get_groundtruth(c))
            print(f"Max_cost: {max_cost}, actual_cost: {current_cost}")
            print(f'Last image contains '
                  f'{len(self.dataset.get_groundtruth(active_indices[-1]))} objects'
            )
            print(f'{len(active_indices)} images selected!')

        return active_indices

    def _select_balance(self, last_cycle_active_images):
        """
        Select images such that the classes are as balanced as possible.
        """
        active_indices = {im_id:1 for im_id in range(len(self.dataset)) 
                          if self.dataset.get_img_info(im_id)['file_name'] in last_cycle_active_images}
        
        num_classes = len(self.dataset.get_categories()) # class 0 is background
        num_images_to_select = self.args.budget

        if len(last_cycle_active_images) > 0:
            class_histogram = np.array(list(self.get_class_histogram(last_cycle_active_images).values()))
        else:
            class_histogram = np.zeros(num_classes)

        inactive_id_by_class = self.get_inactive_id_by_class(last_cycle_active_images)
        for cl in inactive_id_by_class.keys():
            inactive_id_by_class[cl] = np.random.permutation(inactive_id_by_class[cl])

        inactive_positions = [0]*num_classes

        for _ in range(num_images_to_select):
            # select one of the class with the least selected images
            cl = np.random.choice(np.where(class_histogram[1:] == class_histogram[1:].min())[0])+1
            # select one inactive image from this class
            _id = -1
            while inactive_positions[cl] < len(inactive_id_by_class[cl]):
                _id = inactive_id_by_class[cl][inactive_positions[cl]]
                inactive_positions[cl] += 1
                if _id not in active_indices:
                    break

            if _id >= 0 and _id not in active_indices:
                active_indices[_id] = 1
                class_histogram[self.dataset.get_groundtruth(_id).get_field('labels').unique()] += 1
            else:
                # if a class is exhausted, ensure that it is not considered again
                class_histogram[cl] = len(self.dataset) + 1
        
        return list(active_indices.keys())

    def select(self):
        print('Loading model and data_loader ...')
        self.load_config()
        self.load_data()
        
        #-----------

        last_cycle_active_images, _ = self.load_last_cycle_active_list()
        self.set_unlabelled_labelled_dataset(last_cycle_active_images)
        if last_cycle_active_images is None:
            last_cycle_active_images = []
        
        print(f'Selecting images using {self.args.unit} as unit ...')
        if self.args.random_variant == 'uniform':
            image_indices = self._select_uniform(last_cycle_active_images)
        elif self.args.random_variant == 'balance':
            image_indices = self._select_balance(last_cycle_active_images)
        else:
            raise ValueError(f'Value of args.random_variant ({self.args.random_variant}) not supported!')
           
        active_images = [self.dataset.get_img_info(idx)['file_name'] for idx in image_indices]
        save_info = {'args': self.args}

        return active_images, image_indices, save_info



