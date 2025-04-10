import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from args import get_args
from utils import init_cfg, get_predictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from tqdm import tqdm
import torch

def process_footage(input_dir, output_dir, predictor, metadata):
    """Process all frames in the input directory and save segmented results"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all frame files
    frame_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.jpg')])
    
    for frame_file in tqdm(frame_files, desc="Processing frames"):
        # Read image
        im = cv2.imread(os.path.join(input_dir, frame_file))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        
        # Get predictions
        outputs = predictor(im)
        
        # Visualize predictions
        v = Visualizer(
            im,
            metadata=metadata,
            scale=0.8,
            instance_mode=ColorMode(1)
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        # Save visualization
        plt.figure(figsize=(12, 12))
        plt.imshow(v.get_image())
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"segmented_{frame_file}"), 
                   bbox_inches='tight', pad_inches=0)
        plt.close()

def main():
    # Get arguments
    args = get_args()
    base_model = args.model
    
    # Initialize configuration
    cfg = init_cfg(base_model)

    # Set device to CPU
    cfg.MODEL.DEVICE = "cpu"
    print("Using CPU")
    
    # # Set device to MPS (Apple M chip GPU)
    # if torch.backends.mps.is_available():
    #     cfg.MODEL.DEVICE = "mps"
    #     print("Using Apple M chip GPU (MPS)")
    # else:
    #     cfg.MODEL.DEVICE = "cpu"
    #     print("MPS not available, falling back to CPU")
    
    # Set output directory to current directory
    cfg.OUTPUT_DIR = "."
    
    # Set the model path to the current directory
    model_path = "model_final_iter3000_batch4_threshold0.7_freeze0.pth"
    # model_path = "model_final_iter3000_batch4_threshold0.7_freeze5.pth"

    confidence_threshold = 0.75
    
    # Get predictor with your trained model
    predictor = get_predictor(cfg, model_path, threshold=confidence_threshold)
    
    # Get metadata for visualization
    new_thing_classes = args.new_thing_classes
    new_stuff_classes = args.new_stuff_classes
    metadata = MetadataCatalog.get("vis_train")
    metadata.thing_classes = new_thing_classes
    
    # Process footage
    input_dir = "world"
    output_dir = "world_segmented"
    process_footage(input_dir, output_dir, predictor, metadata)

if __name__ == "__main__":
    main() 