#------------------------------------------------------------------------------
# Code developed by Huy V. Vo and Oriane Simeoni                              
# INRIA, Valeo.ai                                                             
#------------------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
from wetectron.utils.env import setup_environment  # noqa F401 isort:skip

import os
import argparse
import numpy as np

from loss_sampler import LossSampler
from entropy_sampler import EntropySampler
from box_in_box_sampler import BoxInBoxSampler
from random_sampler import RandomSampler
from coreset_sampler import CoreSetSampler
from kmeans_2plus_init_sampler import Kmeans2PlusSampler, Kmeans2PlusImageSampler

import torch 
from wetectron.utils.comm import is_main_process, synchronize


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True,
                        help='Name of the experiment whose weights are used')
    parser.add_argument('--cycle', type=int, required=True,
                        help='Selection cycle')
    parser.add_argument('--sel_method', '-al', type=str, required=True,
                        help='Selection method')
    parser.add_argument('--budget', '-b', required=True, type=int,
                        default=50, help='Number of images to select')
    parser.add_argument('--prediction_name', '-pred', type=str, 
                        help='Name of the predictions (the folder containing predictions.pth)')
    parser.add_argument('--save_name_prefix', type=str, required=True, 
                        help='Prefix to active list name')
    parser.add_argument('--ver', type=int, required=True,
                        help='Index of the active list to be saved')
    parser.add_argument('--use_cdb', type=lambda x: x.lower()=='true', required=True, 
                        help='Whether to use adversarial training.')
    parser.add_argument('--config_file', '-cf', type=str, required=True, 
                        help='Config file path, e.g., configs/voc/V_16_voc07_huy_active.yaml')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Folder where the active lists are saved')

    parser.add_argument('--score_thresh', type=float, default=0.5, 
                        help='Score threshold deciding if a prediction is high-confident')
    parser.add_argument('--wetectron_output_path', type=str, default='outputs', 
                        help='Path to wetectron\'s output folder')
    parser.add_argument('--save_active_list', action='store_true', 
                        help='Whether to save the generated active list')
    parser.add_argument('--visualize', action='store_true', 
                        help='Whether to visualize the selected images.')
    parser.add_argument('--vis_path', type=str, default='tmp', help='Visualization path')
    parser.add_argument('--bib_variant', type=int, default=1)
    parser.add_argument('--bib_min_area_ratio', type=float, default=3.0)
    parser.add_argument('--random_variant', type=str, default='balance')
    parser.add_argument('--bib_loss_method', type=str, default="", 
                        help="Loss method used to compute BiB pair weights")
    parser.add_argument('--unit', type=str, default="images", 
                        help="Whether the budget is in image or box")
    parser.add_argument('--bib_pca_dim', type=int, default=0, help='BiB PCA dimension')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    print('Arguments:', args)

    assert(args.unit in ['images', 'boxes'])

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1
    args.__dict__['distributed'] = distributed
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="gloo", init_method="env://"
        )
        synchronize()

    # Select the sampler
    if "entropy" in args.sel_method:
        sampler = EntropySampler(args)
    elif args.sel_method == "BiB":
        sampler = BoxInBoxSampler(args)
    elif "loss" in args.sel_method:
        sampler = LossSampler(args)
    elif "coreset" in args.sel_method:
        sampler = CoreSetSampler(args)
    elif args.sel_method == "random":
        sampler = RandomSampler(args)
    elif args.sel_method == "kmeans2plus":
        sampler = Kmeans2PlusSampler(args)
    elif args.sel_method == "kmeans2plusimage":
        sampler = Kmeans2PlusImageSampler(args)
    else:
        raise ValueError("Unknown selection method.")

    active_dir, active_path = sampler.get_active_list_path()
    if os.path.exists(active_path):
        print(f'Selection {active_path} already exists.')
    else:
        # Apply selection
        active_images, image_indices, save_info = sampler.select()
        if distributed and not is_main_process():
            exit()

        # Print distribution of classes in selection
        print(sampler.get_class_histogram(active_images))

        # Sanity checks
        if args.unit == 'images':
            if len(np.unique(active_images)) != args.cycle * args.budget:
                raise ValueError(
                    f'Nb images incorrect: '
                    f'{len(np.unique(active_images))} != {args.cycle * args.budget}'
                )

            if sampler._previous_cycle_images is not None and \
            len(set(active_images)-set(sampler._previous_cycle_images)) != args.budget:
                raise ValueError("Unexpected overlap of data.")

        # Visualize
        if args.visualize:
            sampler.visualize_active_images(image_indices, sampler.preds)

        # Store selection
        if args.save_active_list:
            sampler.save_active_list(active_images, save_info)
