#------------------------------------------------------------------------------
# Code developed by Huy V. Vo and Oriane Simeoni                              
# INRIA, Valeo.ai                                                             
#------------------------------------------------------------------------------

import torch

import numpy as np
from tqdm import tqdm
from wetectron.structures.image_list import to_image_list
from pathlib import Path

from active_sampler import BaseSampler


class LossSampler(BaseSampler):
    """
    Select images use wetectron losses.
    """

    def __init__(self, *args):
        super().__init__(*args)

        self.use_losses = []

        if 'avg' in self.args.sel_method:
            self.pool = 'avg'
        elif 'mul' in self.args.sel_method:
            self.pool = 'mul'
        elif 'sum' in self.args.sel_method:
            self.pool = 'sum'
        else:
            raise ValueError("Unknown pooling.")

        self.branches = []
        
        if 'cls0' in self.args.sel_method:
            self.branches.append(0)
            self.use_losses.append('loss_ref_cls0')
        if 'cls1' in self.args.sel_method:
            self.branches.append(1)
            self.use_losses.append('loss_ref_cls1')
        if 'cls2' in self.args.sel_method:
            self.branches.append(2)
            self.use_losses.append('loss_ref_cls2')

        if 'reg0' in self.args.sel_method:
            self.branches.append(0)
            self.use_losses.append('loss_ref_reg0')
        if 'reg1' in self.args.sel_method:
            self.branches.append(1)
            self.use_losses.append('loss_ref_reg1')
        if 'reg2' in self.args.sel_method:
            self.branches.append(2)
            self.use_losses.append('loss_ref_reg2')

        if 'img' in self.args.sel_method:
            self.use_losses.append('loss_img')


    def extract_loss_iter(self, images, targets, rois):
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
            if self.model_cdb is not None:
                print("What to do with model_cdb")
            #     raise ValueError("What to do")
            assert(not self.model.roi_heads.training)
            roi_feats  = self.model.roi_heads.go_through_cdb(features, t_rois, self.model_cdb, is_active=False)
            cls_score, det_score, ref_scores, ref_bbox_preds = self.model.roi_heads.predictor.forward_no_softmax(roi_feats, t_rois)
            
            loss_img, accuracy_img = self.model.roi_heads.weak_loss_evaluator(
                                        [cls_score], [det_score], ref_scores, ref_bbox_preds, t_rois, targets
                                    )

        return loss_img 
        
    def extract_losses(self, dataset):
        """
        rois: external boxes
        """
        timer = None # Timer()
        losses = [None] * len(dataset)
        for i in tqdm(range(len(dataset))):
            image, target, roi, image_id = dataset[i]
            losses[i] = self.extract_loss_iter(
                                [image], [target], 
                                [roi]
                            )
        return losses

    def load_losses(self):
        losses = torch.load(Path(
                            self.args.exp_name, "inference",
                            self.args.prediction_name, "losses.pth"
                        ))
        losses = [losses[lidx] for lidx in self.unlabelled_idxs]
        self.losses = losses

    def pool_losses(self):
        print(f'Pooling with {self.pool}')
        if self.pool == 'sum':
            values = np.zeros((len(self.losses)))
            for l_type in self.use_losses:
                v = np.array([l[l_type].cpu().numpy() for l in self.losses])
                values += v
        elif self.pool == 'mul':
            values = np.ones((len(self.losses)))
            for l_type in self.use_losses:
                v = np.array([l[l_type].cpu().numpy() for l in self.losses])
                values *= v
        elif self.pool == 'avg':
            per_b = [[]]*3
            for b in np.unique(self.branches):
                use_loss = [l for l in self.use_losses if str(b) in l]
                for l_type in use_loss:
                    if len(per_b[b]) == 0:
                        per_b[b] = np.array([l[l_type].cpu().numpy() for l in self.losses])
                    else:
                        per_b[b] *= np.array([l[l_type].cpu().numpy() for l in self.losses])
            # Get values per branch
            values_per_b = [p for p in per_b if len(p)>0]
            if len(values_per_b) == 1:
                raise ValueError("Should have several branches when using avg pooling.")
            values = np.array(values_per_b)
            values = np.mean(values, axis=0)
        return values

    def select(self): 
        print(f'Select images based on the loss with {self.args.sel_method}.')
        
        print('Loading model and data_loader ...')
        self.load_config()
        self.load_data()
        self.load_model()

        print('Load already selected images.')
        last_cycle_active_images, last_cycle_active_images_info = self.load_last_cycle_active_list()
        self.set_unlabelled_labelled_dataset(last_cycle_active_images)

        self.load_losses()

        values = self.pool_losses()
        # Select images with the highest loss
        sorted_val = np.argsort(-np.asarray(values)) # Descending order
        if self.args.unit == "images":
            u_indices = sorted_val[:self.args.budget]
        elif self.args.unit == "boxes":
            current_cost = np.sum([len(self.dataset.get_groundtruth(im_id))
                                   for im_id in self.labelled_idxs])
            max_cost = self.args.budget * self.args.cycle
            u_indices = []
            running_idx = 0
            while current_cost < max_cost:
                im_id = self.unlabelled_idxs[sorted_val[running_idx]]
                u_indices.append(running_idx)
                current_cost += len(self.dataset.get_groundtruth(im_id))
                running_idx += 1
            print(f"Max_cost: {max_cost}, actual_cost: {current_cost}")
            if len(u_indices) > 0:
                print(f'Last image contains '
                    f'{len(self.dataset.get_groundtruth(im_id))} objects'
                )
            print(f'{len(u_indices) + len(self.labelled_idxs)} images selected!')
            

        # Get the list of image names
        image_indices = self.unlabelled_idxs[u_indices] # Got back to full data indexing
        active_images = [self.dataset.get_img_info(idx)['file_name'] for idx in image_indices]

        # Save info
        save_info = {
            'sel_method': self.args.sel_method,
            'used_losses': values[u_indices],
            'losses': self.losses,
            'args': self.args
        }

        # Sum to previous selection
        if int(self.args.cycle) > 1: # cycle in base 1
            image_indices = np.append(self.labelled_idxs, image_indices)
            active_images = self._previous_cycle_images + active_images

        return active_images, image_indices, save_info
