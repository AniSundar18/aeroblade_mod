from PIL import Image
import os

def resize_images(source_dir, dest_dir, width, height):
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Walk through the source directory and process image files
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                image_path = os.path.join(root, file)
                
                # Open and resize the image
                with Image.open(image_path) as img:
                    resized_img = img.resize((width, height), Image.LANCZOS)
                    
                    # Construct the destination path
                    relative_path = os.path.relpath(image_path, source_dir)
                    dest_path = os.path.join(dest_dir, relative_path)
                    
                    # Create subdirectories in the destination directory if needed
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    
                    # Save the image in the original format
                    resized_img.save(dest_path)

    print(f"All images have been resized to {width}x{height} pixels and saved to {dest_dir}.")

# Define source and destination directories and desired dimensions
source_directory = '/nobackup3/anirudh/datasets/whichfaceisreal/test_cnn/0_real'
destination_directory = '/nobackup3/anirudh/datasets/whichfaceisreal/test_cnn/0_real_resized'
desired_width = 512
desired_height = 512

# Call the function
resize_images(source_directory, destination_directory, desired_width, desired_height)

