import os
import json
import random
import shutil

def split_dataset(train_dir='dataset/train', val_dir='dataset/val', val_ratio=0.2):
    # Create validation directory if it doesn't exist
    os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
    
    # Load the training annotations
    with open(os.path.join(train_dir, 'via_region_data.json'), 'r') as f:
        annotations = json.load(f)
    
    # Get all image filenames
    image_files = [v['filename'] for v in annotations.values()]
    
    # Calculate number of validation images
    num_val = int(len(image_files) * val_ratio)
    
    # Randomly select validation images
    val_files = random.sample(image_files, num_val)
    
    # Create validation annotations
    val_annotations = {}
    train_annotations = {}
    
    for key, value in annotations.items():
        if value['filename'] in val_files:
            val_annotations[key] = value
            # Move image to validation directory
            src = os.path.join(train_dir, 'images', value['filename'])
            dst = os.path.join(val_dir, 'images', value['filename'])
            shutil.move(src, dst)
        else:
            train_annotations[key] = value
    
    # Save the split annotations
    with open(os.path.join(train_dir, 'via_region_data.json'), 'w') as f:
        json.dump(train_annotations, f)
    
    with open(os.path.join(val_dir, 'via_region_data.json'), 'w') as f:
        json.dump(val_annotations, f)
    
    print(f"Split complete:")
    print(f"Training images: {len(train_annotations)}")
    print(f"Validation images: {len(val_annotations)}")

if __name__ == "__main__":
    split_dataset() 