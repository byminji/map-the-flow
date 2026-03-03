import copy
import logging
import os
import os.path as osp
from os.path import join

import torch
from torch.utils.data import ConcatDataset, DataLoader

from utils.optimizer import create_optimizer
from utils.scheduler import create_scheduler

logger = logging.getLogger(__name__)


# LLaVA special tokens
LLAVA_IMAGE_TOKEN = '<image>'

# InternVL special tokens
INTERNVL_IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
INTERNVL_IMG_START_TOKEN = '<img>'
INTERNVL_IMG_END_TOKEN = '</img>'
INTERNVL_QUAD_START_TOKEN = '<quad>'
INTERNVL_QUAD_END_TOKEN = '</quad>'
INTERNVL_REF_START_TOKEN = '<ref>'
INTERNVL_REF_END_TOKEN = '</ref>'
INTERNVL_BOX_START_TOKEN = '<box>'
INTERNVL_BOX_END_TOKEN = '</box>'

# Normalization factor
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_media_types(datasources):
    """get the media types for for all the dataloaders.

    Args:
        datasources (List): List of dataloaders or datasets.

    Returns: List. The media_types.

    """
    if isinstance(datasources[0], DataLoader):
        datasets = [dataloader.dataset for dataloader in datasources]
    else:
        datasets = datasources
    media_types = [
        dataset.datasets[0].media_type
        if isinstance(dataset, ConcatDataset)
        else dataset.media_type
        for dataset in datasets
    ]

    return media_types
