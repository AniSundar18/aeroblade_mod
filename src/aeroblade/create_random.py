from aeroblade.utils import RandomAugment
import os
import argparse
from PIL import Image
import torch
from torchvision import transforms

def apply_transform(input_folder, output_folder, transform):
    # Create the output directory if it does not exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each image in the input directory
    for filename in os.listdir(input_folder)[:1000]:
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            # Open the image
            image = Image.open(input_path)
            
            # Apply the transform
            transformed_image = transforms.ToPILImage()(transform(image))
            
            # Save the transformed image
            transformed_image.save(output_path)
            print(f"Transformed {input_path} and saved to {output_path}")

def main(input_folder="/nobackup3/anirudh/datasets/SDv2-1/gustavosta/res-512/guidance-7-5/images/1_fake", output_folder="/nobackup3/anirudh/datasets/SDv2-1/gustavosta/res-512/guidance-7-5/images/1_fake_random"):
    transform = RandomAugment()

    # Apply the transform to all images in the input folder
    apply_transform(input_folder, output_folder, transform)

if __name__ == "__main__":
    main()

