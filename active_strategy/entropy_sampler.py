#------------------------------------------------------------------------------
# Code developed by Huy V. Vo and Oriane Simeoni                              
# INRIA, Valeo.ai                                                             
#------------------------------------------------------------------------------

import argparse
import torch

import numpy as np
from tqdm import tqdm
from scipy.stats import entropy as ent
from wetectron.structures.image_list import to_image_list

from active_sampler import BaseSampler
from pathlib import Path


class EntropySampler(BaseSampler):
    """
    Select images based on the entropy of the boxes
    """

    def __init__(self, *args):
        super().__init__(*args)

        # Discard background class
        self.no_bgd_cls = False
        if 'nogd' in self.args.sel_method:
            self.no_bgd_cls = True

        if 'max' in self.args.sel_method:
            self.agg_over_boxes = "max"
        elif 'mean' in self.args.sel_method:
            self.agg_over_boxes = "mean"
        elif 'sum' in self.args.sel_method:
            self.agg_over_boxes = "sum"
        else:
            raise ValueError("Select aggregation scheme.")

    def extract_preds_iter(self, images, targets, rois):
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
        t_rois = [r.to(self.device) if r is not None else None for r in t_rois]

        with torch.no_grad():
            features = self.model.backbone(t_images.to('cuda').tensors)
            # if self.model_cdb is not None:
            #     raise ValueError("What to do")
            x, result, detector_losses, accuracy = self.model.roi_heads(features, t_rois, targets, self.model_cdb)
            
            # Filter with NMS
            filter_results = self.model.roi_heads.strong_post_processor.filter_results
            preds = [filter_results(bbx, self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES) for bbx in result]

            # # Change in order to have training losses
            # # self.model.training = True
            # # self.model.return_all = True
            # # self.model.roi_heads.return_all = True
            # output = self.model(t_images.to('cuda'), targets, rois=t_rois)
        return preds

    def extract_preds(self, dataset):
        """
        rois: external boxes
        """
        timer = None # Timer()
        preds = [None] * len(dataset)
        for i in tqdm(range(len(dataset))):
            image, target, roi, image_id = dataset[i]
            preds[i] = self.extract_preds_iter(
                                [image], [target], 
                                [roi]
                            )[0]
            # if i > 40:
            #     break
            # # boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape)
            # boxlist = boxlist.clip_to_image(remove_empty=False)
            # bbox = torch.mean(torch.stack([boxlist_t.bbox for boxlist_t in boxlist_ts]) ,  dim=0)
            # scores = torch.mean(torch.stack([boxlist_t.get_field('scores') for boxlist_t in boxlist_ts]), dim=0)
            # preds = self.post_processor.filter_results(result, self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES)
            #self.model.roi_heads.weak_post_processor(x,results)
        return preds

    def select(self): 
        print(f'Select images based on the entropy with {self.agg_over_boxes} pooling.')
        if self.no_bgd_cls:
            print(f'Discard back-ground class.')
        
        print('Loading model and data_loader ...')
        self.load_config()
        self.load_data()
        self.load_model()

        print('Load already selected images.')
        last_cycle_active_images, last_cycle_active_images_info = self.load_last_cycle_active_list()
        self.set_unlabelled_labelled_dataset(last_cycle_active_images)

        # Extract predictions
        prediction_path = Path(self.args.exp_name, 'inference',
                               self.args.prediction_name, 'predictions.pth')
        print(f'Loading predictions from {str(prediction_path)} ...')
        predictions = torch.load(prediction_path)
        assert len(predictions) == len(self.dataset), \
            f'Prediction length {len(predictions)} must be the same as dataset length {len(self.dataset)}!'

        preds = [predictions[im_id] for im_id in self.unlabelled_idxs]
        probs = [p.get_field("scores_all").cpu().numpy() for p in preds if p is not None]
        assert(len(probs) == len(preds))

        # Store for visualization
        self.preds = preds

        # Compute entropy per class, over all classes or all but background class
        if not self.no_bgd_cls:
            entropies = [ent(p, axis=1) for p in probs]
        else:
            entropies = [ent(p[:,1:]/p[:,1:].sum(axis=1, keepdims=True), axis=1) for p in probs]

        # Aggregate box scores per image using max or mean 
        if self.agg_over_boxes == "max":
            values = [np.max(ent) for ent in entropies]
        elif self.agg_over_boxes == "mean":
            values = [np.mean(ent) for ent in entropies]
        elif self.agg_over_boxes == "sum":
            values = [np.sum(ent) for ent in entropies]
        
        # Select images with the highest entropy
        values = np.array(values)
        sorted_val = np.argsort(-np.asarray(values)) # Descending order
        u_indices = sorted_val[:self.args.budget]

        # Get the list of image names
        image_indices = self.unlabelled_idxs[u_indices] # Got back to full data indexing
        active_images = [self.dataset.get_img_info(idx)['file_name'] for idx in image_indices]

        # Save info
        save_info = {
            'entropy': values[u_indices],
            'agg': self.agg_over_boxes,
            'no_bgd_cls': self.no_bgd_cls,
            'args': self.args
        }

        # Sum to previous selection
        if int(self.args.cycle) > 1: # cycle in base 1
            image_indices = np.append(self.labelled_idxs, image_indices)
            active_images = self._previous_cycle_images + active_images

        return active_images, image_indices, save_info