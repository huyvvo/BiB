#------------------------------------------------------------------------------
# Code adapted from https://github.com/NVlabs/wetectron
# by Huy V. Vo and Oriane Simeoni
# INRIA, Valeo.ai
#------------------------------------------------------------------------------

# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# --------------------------------------------------------
import torch
from torch import nn
from torch.nn import functional as F

import pdb

from ..box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
from ..box_head.loss import make_roi_box_loss_evaluator
from ..box_head.inference import make_roi_box_post_processor as strong_roi_box_post_processor

from .roi_weak_predictors import make_roi_weak_predictor
from .inference import make_roi_box_post_processor as weak_roi_box_post_processor
from .loss import make_roi_weak_loss_evaluator, generate_img_label, make_active_weak_loss_evaluator
from .roi_sampler import make_roi_sampler

from wetectron.modeling.utils import cat
from wetectron.structures.boxlist_ops import cat_boxlist
from wetectron.structures.bounding_box import BoxList


class ROIWeakHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIWeakHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_weak_predictor(cfg, self.feature_extractor.out_channels)
        self.post_processor = weak_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_weak_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None, model_cdb=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """        
        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        cls_score, det_score, ref_scores = self.predictor(x, proposals)
        if not self.training:
            if ref_scores == None:
                final_score = cls_score * det_score
            else:
                final_score = torch.mean(torch.stack(ref_scores), dim=0)
            result = self.post_processor(final_score, proposals)
            return x, result, {}, {}

        loss_img, accuracy_img = self.loss_evaluator([cls_score], [det_score], ref_scores, proposals, targets)
        
        return (
            x,
            proposals,
            loss_img,
            accuracy_img
        )


class ROIWeakRegHead(torch.nn.Module):
    """ Generic Box Head class w/ regression. """
    def __init__(self, cfg, in_channels):
        super(ROIWeakRegHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_weak_predictor(cfg, self.feature_extractor.out_channels)
        self.weak_loss_evaluator = make_roi_weak_loss_evaluator(cfg)
        
        self.weak_post_processor = weak_roi_box_post_processor(cfg)
        self.strong_post_processor = strong_roi_box_post_processor(cfg)
        
        self.HEUR = cfg.MODEL.ROI_WEAK_HEAD.REGRESS_HEUR
        self.roi_sampler = make_roi_sampler(cfg) if cfg.MODEL.ROI_WEAK_HEAD.PARTIAL_LABELS != "none" else None
        self.DB_METHOD = cfg.DB.METHOD

        self.return_loss = cfg.TEST.RETURN_LOSS

    def go_through_cdb(self, features, proposals, model_cdb):
        if not self.training or self.DB_METHOD == "none":
            return self.feature_extractor(features, proposals)
        elif self.DB_METHOD == "concrete":
            x = self.feature_extractor.forward_pooler(features, proposals)
            x = model_cdb(x)
            return self.feature_extractor.forward_neck(x)
        else:
            raise ValueError
        
    def forward(self, features, proposals, targets=None, model_cdb=None):
        # for partial labels
        if self.roi_sampler is not None and self.training:
            with torch.no_grad():
                proposals = self.roi_sampler(proposals, targets)
        roi_feats  = self.go_through_cdb(features, proposals, model_cdb)
        cls_score, det_score, ref_scores, ref_bbox_preds = self.predictor.forward_no_softmax(roi_feats, proposals)
        if not self.training:
            # Compute predictions
            cls_score_softmax = F.softmax(cls_score, dim=1)
            # do softmax along ROI for different imgs
            det_logit_list = det_score.split([len(p) for p in proposals])
            det_logit_softmax = []
            for det_logit_per_image in det_logit_list:
                det_logit_softmax.append(F.softmax(det_logit_per_image, dim=0))
            det_score_softmax = torch.cat(det_logit_softmax, dim=0)
            ref_scores_softmax = [F.softmax(_ref, dim=1) for _ref in ref_scores]
            
            result = self.testing_forward(cls_score_softmax, det_score_softmax, proposals, 
                                          ref_scores_softmax, ref_bbox_preds)
            if self.return_loss:
                loss_img, _ = self.weak_loss_evaluator.forward_per_im(
                                [cls_score], [det_score], 
                                ref_scores, ref_bbox_preds, 
                                proposals, targets
                            )
                result = zip(result, loss_img)

            return roi_feats, result, {}, {}
        loss_img, accuracy_img = self.weak_loss_evaluator([cls_score], [det_score], ref_scores, ref_bbox_preds, proposals, targets)
        return (roi_feats, proposals, loss_img, accuracy_img)

    def testing_forward(self, cls_score, det_score, proposals, ref_scores=None, ref_bbox_preds=None):
        if self.HEUR == "WSDDN":
            final_score = cls_score * det_score
            result = self.weak_post_processor(final_score, proposals)
        elif self.HEUR == "CLS-AVG":
            final_score = torch.mean(torch.stack(ref_scores), dim=0)
            result = self.weak_post_processor(final_score, proposals)
        elif self.HEUR == "AVG": # AVG
            final_score = torch.mean(torch.stack(ref_scores), dim=0)
            final_regression = torch.mean(torch.stack(ref_bbox_preds), dim=0)
            result = self.strong_post_processor((final_score, final_regression), proposals, softmax_on=False)
        elif self.HEUR == "UNION": # UNION
            prop_list = [len(p) for p in proposals]
            ref_score_list = [rs.split(prop_list) for rs in ref_scores]
            ref_bbox_list = [rb.split(prop_list) for rb in ref_bbox_preds]
            final_score = [torch.cat((ref_score_list[0][i], ref_score_list[1][i], ref_score_list[2][i])) for i in range(len(proposals)) ]
            final_regression = [torch.cat((ref_bbox_list[0][i], ref_bbox_list[1][i], ref_bbox_list[2][i])) for i in range(len(proposals)) ]
            augmented_proposals = [cat_boxlist([p for _ in range(3)]) for p in proposals]
            result = self.strong_post_processor((cat(final_score), cat(final_regression)), augmented_proposals, softmax_on=False)
        else:
            raise ValueError
        return result

class ROIWeakRegHeadActive(torch.nn.Module):
    """ Generic Box Head class w/ regression. """
    def __init__(self, cfg, in_channels):
        super(ROIWeakRegHeadActive, self).__init__()

        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_weak_predictor(cfg, self.feature_extractor.out_channels)
        self.weak_loss_evaluator = make_roi_weak_loss_evaluator(cfg)
        self.active_loss_evaluator = make_active_weak_loss_evaluator(cfg)
        self.active_loss_weight = cfg.MODEL.ROI_WEAK_HEAD.ACTIVE_LOSS_WEIGHT
        self.active_bbx_loss_weight = cfg.ACTIVE.WEIGHTS_BBX_LOSS
        self.active_loss_img_strong_det = cfg.ACTIVE.IMG_STRONG_DET_WEIGHT
        
        self.weak_post_processor = weak_roi_box_post_processor(cfg)
        self.strong_post_processor = strong_roi_box_post_processor(cfg)
        
        self.HEUR = cfg.MODEL.ROI_WEAK_HEAD.REGRESS_HEUR
        self.roi_sampler = make_roi_sampler(cfg) if cfg.MODEL.ROI_WEAK_HEAD.PARTIAL_LABELS != "none" else None
        self.DB_METHOD = cfg.DB.METHOD
        self.return_loss = cfg.TEST.RETURN_LOSS

        self.weighted_proposal_subsample = cfg.ACTIVE.WEIGHTED_PROPOSAL_SUBSAMPLE
        self.active_go_through_cdb = cfg.ACTIVE.GO_THROUGH_CDB


    def go_through_cdb(self, features, proposals, model_cdb, is_active):
        if not self.training or self.DB_METHOD == "none":
            return self.feature_extractor(features, proposals)
        elif self.DB_METHOD == "concrete":
            if is_active and not self.active_go_through_cdb:
                return self.feature_extractor(features, proposals)
            else:
                x = self.feature_extractor.forward_pooler(features, proposals)
                x = model_cdb(x)
                return self.feature_extractor.forward_neck(x)
        else:
            raise ValueError

    def get_loss(self, features, proposals, targets=None, model_cdb=None, epsilon=1e-10, return_score=False):
        assert(len(targets) == 1)
        is_active = targets is not None and targets[0].has_field('active')
        assert(not self.training)
        roi_feats  = self.go_through_cdb(features, proposals, model_cdb, is_active)
        cls_score, det_score, ref_scores, ref_bbox_preds = self.predictor.forward_no_softmax(roi_feats, proposals)
        
        if is_active:
            loss_img, accuracy_img = self.active_loss_evaluator(
                                        [cls_score], [det_score], ref_scores, ref_bbox_preds, proposals, targets
                                    )
            loss_img['loss_img_strong_det'] *= self.active_bbx_loss_weight * self.active_loss_img_strong_det
            loss_img['loss_ref_reg0'] *= self.active_bbx_loss_weight
            loss_img['loss_ref_reg1'] *= self.active_bbx_loss_weight
            loss_img['loss_ref_reg2'] *= self.active_bbx_loss_weight

            for k in loss_img:
                loss_img[k] *= self.active_loss_weight * targets[0].get_field('active_instance_weight')
        else:
            loss_img, accuracy_img = self.weak_loss_evaluator(
                                        [cls_score], [det_score], ref_scores, ref_bbox_preds, proposals, targets
                                    )
            device = roi_feats.device
            loss_img['loss_img_strong_class'] = torch.tensor(0.0).to(device)
            loss_img['loss_img_strong_det'] = torch.tensor(0.0).to(device)
        if return_score:
            return (loss_img, cls_score, det_score, ref_scores, ref_bbox_preds)
        else:
            return (loss_img,)
        
    def forward(self, features, proposals, targets=None, model_cdb=None, epsilon=1e-10):
        is_active = targets is not None and targets[0].has_field('active')
        if self.training:
            with torch.no_grad():
                if is_active:
                    if self.weighted_proposal_subsample:
                        roi_feats  = self.go_through_cdb(features, proposals, model_cdb, is_active)
                        cls_score, det_score, ref_scores, ref_bbox_preds = self.predictor(roi_feats, proposals)
                        ref_scores = F.softmax(torch.mean(torch.stack(ref_scores), dim=0), dim=1)
                        sampling_weights = torch.max(ref_scores[:,1:], dim=1)[0]
                        sampling_weights = sampling_weights.split([len(p) for p in proposals])
                    else:
                        sampling_weights=None

                    # Faster R-CNN subsamples during training the proposals with a fixed
                    # positive / negative ratio
                    # Match proposals to targets as positive or negative
                    proposals = self.active_loss_evaluator.subsample(proposals, targets, sampling_weights)
                elif self.roi_sampler is not None:
                    with torch.no_grad():
                        proposals = self.roi_sampler(proposals, targets)
                
        # Get features per RoI
        roi_feats  = self.go_through_cdb(features, proposals, model_cdb, is_active)
        cls_score, det_score, ref_scores, ref_bbox_preds = self.predictor.forward_no_softmax(roi_feats, proposals)
        if not self.training:
            # Compute predictions
            cls_score_softmax = F.softmax(cls_score, dim=1)
            # do softmax along ROI for different imgs
            det_logit_list = det_score.split([len(p) for p in proposals])
            det_logit_softmax = []
            for det_logit_per_image in det_logit_list:
                det_logit_softmax.append(F.softmax(det_logit_per_image, dim=0))
            det_score_softmax = torch.cat(det_logit_softmax, dim=0)
            ref_scores_softmax = [F.softmax(_ref, dim=1) for _ref in ref_scores]
            
            result = self.testing_forward(cls_score_softmax, det_score_softmax, proposals, 
                                          ref_scores_softmax, ref_bbox_preds)
            if self.return_loss:
                loss_img, _ = self.weak_loss_evaluator.forward_per_im(
                                [cls_score], [det_score], 
                                ref_scores, ref_bbox_preds, 
                                proposals, targets
                            )
                result = zip(result, loss_img)

            return roi_feats, result, {}, {}
        
        if is_active:
            loss_img, accuracy_img = self.active_loss_evaluator(
                                        [cls_score], [det_score], ref_scores, ref_bbox_preds, proposals, targets
                                    )
            loss_img['loss_img_strong_det'] *= self.active_bbx_loss_weight * self.active_loss_img_strong_det
            loss_img['loss_ref_reg0'] *= self.active_bbx_loss_weight
            loss_img['loss_ref_reg1'] *= self.active_bbx_loss_weight
            loss_img['loss_ref_reg2'] *= self.active_bbx_loss_weight

            for k in loss_img:
                loss_img[k] *= self.active_loss_weight * targets[0].get_field('active_instance_weight')
        else:
            loss_img, accuracy_img = self.weak_loss_evaluator(
                                        [cls_score], [det_score], ref_scores, ref_bbox_preds, proposals, targets
                                    )
            device = roi_feats.device
            loss_img['loss_img_strong_class'] = torch.tensor(0.0).to(device)
            loss_img['loss_img_strong_det'] = torch.tensor(0.0).to(device)

        return (roi_feats, proposals, loss_img, accuracy_img)

    def testing_forward(self, cls_score, det_score, proposals, ref_scores=None, ref_bbox_preds=None):
        if self.HEUR == "WSDDN":
            final_score = cls_score * det_score
            final_regression = torch.zeros(final_score.shape[0], final_score.shape[1]*4) # dummy variables
            result = self.weak_post_processor(final_score, proposals)
        elif self.HEUR == "CLS-AVG":
            final_score = torch.mean(torch.stack(ref_scores), dim=0)
            result = self.weak_post_processor(final_score, proposals)
        elif self.HEUR == "AVG": # AVG
            final_score = torch.mean(torch.stack(ref_scores), dim=0)
            final_regression = torch.mean(torch.stack(ref_bbox_preds), dim=0)
            result = self.strong_post_processor((final_score, final_regression), proposals, softmax_on=False)
        elif self.HEUR == "UNION": # UNION
            prop_list = [len(p) for p in proposals]
            ref_score_list = [rs.split(prop_list) for rs in ref_scores]
            ref_bbox_list = [rb.split(prop_list) for rb in ref_bbox_preds]
            final_score = [torch.cat((ref_score_list[0][i], ref_score_list[1][i], ref_score_list[2][i])) for i in range(len(proposals)) ]
            final_regression = [torch.cat((ref_bbox_list[0][i], ref_bbox_list[1][i], ref_bbox_list[2][i])) for i in range(len(proposals)) ]
            augmented_proposals = [cat_boxlist([p for _ in range(3)]) for p in proposals]
            result = self.strong_post_processor((cat(final_score), cat(final_regression)), augmented_proposals, softmax_on=False)
        else:
            raise ValueError
        return result
        
def build_roi_weak_head(cfg, in_channels):
    """
    Constructs a new weak head.
    By default, uses ROIWeakRegHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    if cfg.MODEL.ROI_WEAK_HEAD.ACTIVE_LOSS:
        return ROIWeakRegHeadActive(cfg, in_channels)
    elif cfg.MODEL.ROI_WEAK_HEAD.REGRESS_ON:
        return ROIWeakRegHead(cfg, in_channels)
    else:
       return RoiRegHead(cfg, in_channels)