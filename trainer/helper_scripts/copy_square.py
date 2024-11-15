import os
import shutil
from PIL import Image

def copy_512x512_images(src_directory, dest_directory):
    # Ensure the destination directory exists
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    
    for root, _, files in os.walk(src_directory):
        for file in files:
            print(file)
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    if img.size == (512, 512):
                        # Construct the new file path
                        relative_path = os.path.relpath(root, src_directory)
                        dest_path = os.path.join(dest_directory, relative_path)
                        if not os.path.exists(dest_path):
                            os.makedirs(dest_path)
                        
                        # Copy the file to the destination directory
                        shutil.copy(file_path, os.path.join(dest_path, file))
            except (IOError, SyntaxError) as e:
                # Skip files that cannot be opened as images
                print(f"Skipping file {file_path} due to error: {e}")

# Example usage:
src_directory = '/nobackup3/anirudh/datasets/wikiart/Pointillism'
dest_directory = '/nobackup3/anirudh/datasets/wiki_pointillism_square'
copy_512x512_images(src_directory, dest_directory)

