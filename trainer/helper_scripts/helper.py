import os
import shutil
from PIL import Image

def cluster_images_by_resolution(src_directory, dest_directory):
    # Ensure the destination directory exists
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    
    for root, _, files in os.walk(src_directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    # Get image resolution
                    resolution = f"{img.width}x{img.height}"
                    
                    # Construct the destination path based on resolution
                    resolution_dir = os.path.join(dest_directory, resolution)
                    if not os.path.exists(resolution_dir):
                        os.makedirs(resolution_dir)
                    
                    # Copy the file to the destination directory
                    shutil.copy(file_path, os.path.join(resolution_dir, file))
            except (IOError, SyntaxError) as e:
                # Skip files that cannot be opened as images
                print(f"Skipping file {file_path} due to error: {e}")

# Example usage:
src_directory = '/nobackup3/anirudh/datasets/wikiart'
dest_directory = '/nobackup3/anirudh/datasets/wikiart_clustered'
cluster_images_by_resolution(src_directory, dest_directory)


