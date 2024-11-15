import os
import random
from PIL import Image, ImageEnhance

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def color_jitter(image):
    # Apply random brightness
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.uniform(0.5, 1.5))

    # Apply random contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(random.uniform(0.5, 1.5))

    # Apply random saturation
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(random.uniform(0.5, 1.5))

    # Apply random sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(random.uniform(0.5, 1.5))

    return image

def process_images(input_folder, output_folder):
    create_directory(output_folder)
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff')):
                img_path = os.path.join(root, file)
                img = Image.open(img_path)

                # Apply color jitter
                img = color_jitter(img)

                # Save the processed image to the output folder
                output_path = os.path.join(output_folder, file)
                img.save(output_path)

def main():
    # User inputs
    input_folder = "/nobackup3/anirudh/datasets/SDv1-5/cifar-100/res-512/guidance-7-5/images/1_fake"
    output_folder = "/nobackup3/anirudh/datasets/SDv1-5/cifar-100/res-512/guidance-7-5/images/1_fake_jittered"

    process_images(input_folder, output_folder)
    print(f"Processing complete. Color jittered images saved in: {output_folder}")

if __name__ == "__main__":
    main()

