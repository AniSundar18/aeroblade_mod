import os
import random
import shutil

def copy_random_images(source_dir, dest_dir, num_images):
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Walk through the source directory and collect image paths
    image_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                image_files.append(os.path.join(root, file))
    
    # Check if there are enough images in the source directory
    if len(image_files) < num_images:
        raise ValueError(f"Not enough images in the source directory. Found {len(image_files)}, but need {num_images}.")
    
    # Randomly select the specified number of images
    selected_images = random.sample(image_files, num_images)
    
    # Copy selected images to the destination directory
    for image in selected_images:
        shutil.copy(image, dest_dir)
    
    print(f"Successfully copied {num_images} images to {dest_dir}")

# Define source and destination directories
source_directory = '/nobackup3/anirudh/datasets/dreamsim/dataset/dataset/nights/distort'
destination_directory = '/nobackup3/anirudh/datasets/nights_subset/og'
number_of_images = 1000

# Call the function
copy_random_images(source_directory, destination_directory, number_of_images)

