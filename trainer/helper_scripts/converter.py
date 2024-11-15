import os
import random
from PIL import Image

def convert_images(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all PNG files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.png'):
            input_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{base_name}.jpg")

            # Open the image
            with Image.open(input_path) as img:
                # Generate a random compression quality between 50 and 100
                quality = random.randint(50, 100)
                # Convert and save the image as JPG with the specified quality
                img.convert('RGB').save(output_path, 'JPEG', quality=quality)
                print(f"Converted {input_path} to {output_path} with quality {quality}")

if __name__ == "__main__":
    input_dir = "/nobackup3/anirudh/datasets/nights_subset/og"  # Replace with your input directory path
    output_dir = "/nobackup3/anirudh/datasets/nights_subset/og_jpg"  # Replace with your output directory path
    convert_images(input_dir, output_dir)

