import os
from PIL import Image

def center_crop(image, size):
    width, height = image.size
    new_width, new_height = size

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    return image.crop((left, top, right, bottom))

def process_images(src_directory, dest_directory, crop_size=(512, 512), max_images=1000):
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    
    count = 0
    for root, _, files in os.walk(src_directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    if img.size[0] >= crop_size[0] and img.size[1] >= crop_size[1]:
                        cropped_img = center_crop(img, crop_size)
                        cropped_img_path = os.path.join(dest_directory, file)
                        cropped_img.save(cropped_img_path)
                        count += 1
                        if count >= max_images:
                            return
            except (IOError, SyntaxError) as e:
                print(f"Skipping file {file_path} due to error: {e}")

# Example usage:
src_directory = '/nobackup3/anirudh/datasets/wikiart'
dest_directory = '/nobackup3/anirudh/datasets/wikiart_square'
process_images(src_directory, dest_directory)

