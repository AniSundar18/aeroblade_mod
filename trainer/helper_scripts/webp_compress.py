import os
from PIL import Image

def convert_images_to_webp(input_folder, output_folder):
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  # Add other formats if needed
                input_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_file_path, input_folder)
                output_file_path = os.path.join(output_folder, os.path.splitext(relative_path)[0] + '.webp')

                # Ensure the output subdirectory exists
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

                # Open, convert, and save the image
                with Image.open(input_file_path) as img:
                    img.save(output_file_path, 'webp')

# Usage
input_folder = '/nobackup3/anirudh/datasets/whichfaceisreal/ddim'
output_folder = '/nobackup3/anirudh/datasets/whichfaceisreal/ddim_webp'
convert_images_to_webp(input_folder, output_folder)

