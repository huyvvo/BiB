#------------------------------------------------------------------------------
# Code adapted from https://github.com/NVlabs/wetectron
# by Huy V. Vo and Oriane Simeoni
# INRIA, Valeo.ai
#------------------------------------------------------------------------------

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

import torch
import pickle
from tqdm import tqdm
import pdb
import numpy as np

from wetectron.config import cfg
from wetectron.utils.logger import TqdmToLogger
from wetectron.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from ..utils.visualize import vis_results
from .bbox_aug import im_detect_bbox_aug


def compute_on_dataset(model, data_loader, device, timer=None, vis=False):
    model.eval()
    results_dict = {}
    losses_dict = {}
    cpu_device = torch.device("cpu")
    logger = logging.getLogger(__name__)
    for _, batch in enumerate(tqdm(data_loader,file=TqdmToLogger(logger))):
        images, targets, rois, image_ids = batch
        with torch.no_grad():
            if timer:
                timer.tic()
            if cfg.TEST.BBOX_AUG.ENABLED:
                # TODO: augment with proposal
                assert(not model.roi_heads.return_loss)
                output = im_detect_bbox_aug(model, images, device, rois)
            else:
                targets = [target.to(device) for target in targets]
                rois = [r.to(device) if r is not None else None for r in rois]
                if model.roi_heads.return_loss: 
                    output = model(images.to(device), targets=targets, rois=rois)
                else:
                    output = model(images.to(device), rois=rois)
            if timer:
                if not cfg.MODEL.DEVICE == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            if model.roi_heads.return_loss:
                output = [(o[0].to(cpu_device), {k:v.to(cpu_device) for k,v in o[1].items()})
                              for o in output]
            else:
                output = [o.to(cpu_device) for o in output]
            if vis:
                data_path = data_loader.dataset.root
                img_infos = [data_loader.dataset.get_img_info(ind) for ind in image_ids]
                vis_results(output, img_infos, data_path, show_mask_heatmaps=False)
 
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )

    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("wetectron.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        vis=False,
        task='det',
        run_evaluation=True,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("wetectron.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    if not os.path.exists( os.path.join(output_folder, "predictions.pth")):
        outputs = compute_on_dataset(model, data_loader, device, inference_timer, vis)

        predictions = outputs
        # wait for all processes to complete before measuring the time
        synchronize()
        total_time = total_timer.toc()
        total_time_str = get_time_str(total_time)
        logger.info(
            "Total run time: {} ({} s / img per device, on {} devices)".format(
                total_time_str, total_time * num_devices / len(dataset), num_devices
            )
        )
        total_infer_time = get_time_str(inference_timer.total_time)
        logger.info(
            "Model inference time: {} ({} s / img per device, on {} devices)".format(
                total_infer_time,
                inference_timer.total_time * num_devices / len(dataset),
                num_devices,
            )
        )

        predictions = _accumulate_predictions_from_multiple_gpus(predictions)
        if not is_main_process():
            return

        if output_folder:
            if model.roi_heads.return_loss:
                _pred = [o[0] for o in predictions]
                _loss = [o[1] for o in predictions]
                torch.save(_pred, os.path.join(output_folder, "predictions.pth"))
                torch.save(_loss, os.path.join(output_folder, "losses.pth"))
                predictions = _pred
            else:
                torch.save(predictions, os.path.join(output_folder, "predictions.pth"))
    else:
        predictions = torch.load(os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )
    if run_evaluation:
        return evaluate(dataset=dataset,
                        predictions=predictions,
                        output_folder=output_folder,
                        task=task,
                        **extra_args)
    else:
        print('Return without running evaluation')
    