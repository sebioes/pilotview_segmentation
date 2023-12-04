import os
from tqdm import tqdm
from PIL import Image
import argparse

"""Modified work done by Sriharan Balakrishnan Selvarakumaran"""

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default=None, help='path to the video to be converted to frames')
args = parser.parse_args()

# Directory containing your images
image_dir = args.path

# List all image files in the directory
image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg', '.gif'))])

# Create a list to hold image objects
images = []

# Load each image and append it to the list
for image_file in tqdm(image_files, desc="Loading images"):
    img = Image.open(image_file)
    images.append(img)

# Output GIF file path
output_gif = "output.gif"

# Save the images as a GIF
print("Converting images to GIF...")
images[0].save(output_gif, save_all=True, append_images=images[1:], loop=0, duration=35) # duration is in ms

print(f"GIF saved as {output_gif}")
