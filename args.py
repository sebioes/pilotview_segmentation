import argparse
import os
import sys
sys.path.insert(0, os.path.abspath('./detectron2'))
from detectron2.data import transforms as T

"""Data augmentation for training"""
augmentations = [T.RandomRotation(angle=[-10, 10])]

"""Get arguments (hyperparams)"""
def get_args():
    parser = argparse.ArgumentParser(description='Arguments for training a segmentation model provided by Detectron2')
    
    # Configurations
    parser.add_argument('--data_dir', type=str, default='dataset', help='Custom dataset directory')
    parser.add_argument('--num_gpus_per_machine', type=int, default=1, help='Number of GPUs per machine')
    parser.add_argument('--num_machines', type=int, default=1, help='Number of machines')
    parser.add_argument('--model', type=str, default='COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml', help='Base model')
    parser.add_argument('--new_thing_classes', nargs='+', default=["sky", "field", "road", "building", "river", "monitor"], help='New thing classes')
    parser.add_argument('--new_stuff_classes', nargs='+', default=[], help='New stuff classes')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for minimal confidence for an object to be detected')
    parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.00025, help='Learning rate')
    parser.add_argument('--iter', type=int, default=1500, help='Max training iterations')
    parser.add_argument('--backbone_freeze_at', type=int, default=0, help='Freeze the first several stages of the ResNet. Commonly used in fine-tuning')
    parser.add_argument('--roi_heads_freeze_at', type=int, default=0, help='Freeze the ROI (Region of Interest) heads')
    parser.add_argument('--augmentations', nargs='+', default=augmentations, help='Data augmentations for training')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    print(args.num_gpus_per_machine)
