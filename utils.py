import os
import sys
import json
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from args import get_args

sys.path.insert(0, os.path.abspath('./detectron2'))
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BoxMode

from detectron2.data import DatasetMapper
from detectron2.data import build_detection_train_loader
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode

# Get arguments
args = get_args()
custom_dataset_dir = args.data_dir
vis_dir = args.vis_dir
batch = args.batch
lr = args.lr
iter = args.iter
num_new_classes = len(args.new_thing_classes) + len(args.new_stuff_classes)
threshold = args.threshold
backbone_freeze_at = args.backbone_freeze_at
roi_heads_freeze_at = args.roi_heads_freeze_at
new_classes = args.new_thing_classes + args.new_stuff_classes

def get_data_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            # assert not anno["region_attributes"]
            label = anno["region_attributes"]["label"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                # "category_id": 0,
                "category_id": new_classes.index(label),
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        """Check detectron2/data/transforms.augmentation_impl.py for more augmentations you can use"""
        args = get_args()
        augmentations = args.augmentations
        mapper = DatasetMapper(cfg, is_train=True, augmentations=augmentations) # apply augmentations
        return build_detection_train_loader(cfg, mapper=mapper)

def train(cfg):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer = MyTrainer(cfg) # This trainer contains the data augmentation
    trainer.resume_or_load(resume=False)
    trainer.train()

def init_cfg(config_file: str):
    """Get configuration"""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.DATASETS.TRAIN = (custom_dataset_dir + "_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)  # Let training initialize from model zoo

    # Threshold for our model's minimal confidence for an object to be detected
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

    # Freeze the weights
    """
    Freeze the first several stages of the ResNet. Commonly used in
    fine-tuning.
    Layers that produce the same feature map spatial size are defined as one
    "stage" by :paper:`FPN`.
    Args:
        freeze_at (int): number of stages to freeze.
            `0` means not freezing any parameters.
            `1` means freezing the stem layer. 
            `2` means freezing the stem layer and one residual block, etc.
    """
    cfg.MODEL.BACKBONE.FREEZE_AT = backbone_freeze_at # 0 ~ 5
    cfg.MODEL.ROI_HEADS.FREEZE_AT = roi_heads_freeze_at
    cfg.MODEL.PROPOSAL_GENERATOR.FREEZE_AT = 0

    cfg.SOLVER.IMS_PER_BATCH = batch
    cfg.SOLVER.BASE_LR = lr  
    cfg.SOLVER.MAX_ITER = iter  
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 # (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_new_classes # number of classes

    return cfg

def get_num_params(model):
    """Get the total number of parameters of the model"""
    num_params = sum(p.numel() for p in model.parameters())
    print("Number of parameters: ", num_params)

    """Get the number of trainable parameters of the model"""
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: ", num_trainable_params)

    """Print layers with trainable parameters"""
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"\t{name}")

#################### for Visualization of Infereced Images ####################

def get_predictor(cfg, model_name:str, threshold:float=0.5):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_name)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set the testing threshold for this model
    cfg.DATASETS.TEST = (custom_dataset_dir + "_val", )
    predictor = DefaultPredictor(cfg)
    return predictor

def visualize_prediction(predictor, dataset_type:str="val", num_display:int=3):
    args = get_args()
    new_thing_classes = args.new_thing_classes
    new_stuff_classes = args.new_stuff_classes

    for d in ["train", "val"]:
        DatasetCatalog.register(vis_dir + "_" + d, lambda d=d: get_data_dicts(custom_dataset_dir + "/" + d))
        MetadataCatalog.get(vis_dir + "_" + d).set(thing_classes=new_thing_classes, stuff_classes=new_stuff_classes)
    metadata = MetadataCatalog.get(vis_dir + "_train")

    """Visualize prediction on certain dataset (val or test)"""
    dir = os.path.join(vis_dir, dataset_type)
    count = 0
    for d in random.sample(os.listdir(dir), len(os.listdir(dir))):
        if d[-4:] in [".jpg", ".png", "jpeg"] and count < num_display:
            im = cv2.imread(os.path.join(dir, d))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            outputs = predictor(im)
            v = Visualizer(im,
                        metadata=metadata,
                        scale=0.8,
                        #    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
                        )
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            plt.figure(figsize = (12, 12))
            plt.imshow(v.get_image())

            if not os.path.exists(os.path.join(vis_dir, "vis")):
                os.makedirs(os.path.join(vis_dir, "vis"))
            plt.savefig(os.path.join(vis_dir, "vis", d)) # save the visualized inference

            count += 1