import os
import sys
import json
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

from args import get_args
from utils import get_data_dicts, train, init_cfg, get_num_params

sys.path.insert(0, os.path.abspath('./detectron2'))
import torch
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2.engine import launch
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine.defaults import build_model

# for data augmentation
from detectron2.data import transforms as T

"""Train a segmentation model on a dataset."""
if __name__ == "__main__":
    # Get arguments (hyperparams)
    args = get_args()
    custom_dataset_dir = args.data_dir
    num_gpus_per_machine = args.num_gpus_per_machine
    num_machines = args.num_machines
    base_model = args.model
    batch = args.batch
    lr = args.lr
    iter = args.iter
    new_thing_classes = args.new_thing_classes
    new_stuff_classes = args.new_stuff_classes
    threshold = args.threshold
    backbone_freeze_at = args.backbone_freeze_at
    roi_heads_freeze_at = args.roi_heads_freeze_at
    augmentations = args.augmentations

    # Print versions
    TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
    CUDA_VERSION = torch.__version__.split("+")[-1]
    print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
    print("detectron2:", detectron2.__version__)

    # Print hyperparams
    for arg in vars(args):
        print(arg, getattr(args, arg))

    # New classes
    new_classes = args.new_thing_classes + args.new_stuff_classes

    # Number of new classes
    num_new_classes = len(new_classes)

    for d in ["train", "val"]:
        DatasetCatalog.register(custom_dataset_dir + "_" + d, lambda d=d: get_data_dicts(custom_dataset_dir + "/" + d))
        MetadataCatalog.get(custom_dataset_dir + "_" + d).set(thing_classes=new_thing_classes, stuff_classes=new_stuff_classes)
    metadata = MetadataCatalog.get(custom_dataset_dir + "_train")

    # Retrieve the model
    cfg = init_cfg(base_model)
    model = build_model(cfg)
    
    # Print number of parameters
    get_num_params(model)

    # Train
    launch(
        train,
        num_gpus_per_machine=num_gpus_per_machine,  # Number of GPUs per machine
        num_machines=num_machines,  # Number of machines
        machine_rank=0,
        dist_url=None,
        args=(cfg,),
        )
