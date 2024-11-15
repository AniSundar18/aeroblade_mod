import os
from PIL import Image
import torchvision.transforms as transforms

def resample_image(image, scale_factor=1.5, interpolation=Image.BILINEAR):
    # Upsample the image
    upsample_size = (int(image.width * scale_factor), int(image.height * scale_factor))
    upsampled_image = image.resize(upsample_size, interpolation)

    # Downsample the image back to the original size
    downsampled_image = upsampled_image.resize((image.width, image.height), interpolation)

    return downsampled_image

def process_images(input_folder, output_folder, scale_factor=0.5, interpolation=Image.BILINEAR):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder)[:1000]:
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path)

            # Resample the image
            processed_image = resample_image(image, scale_factor=scale_factor, interpolation=interpolation)

            # Save the processed image
            final_image_path = os.path.join(output_folder, filename)
            processed_image.save(final_image_path)

if __name__ == "__main__":
    input_folder = "/nobackup3/anirudh/datasets/SDv1-5/gustavosta/res-512/guidance-7-5/images/1_fake"  # Replace with your input folder path
    output_folder = "/nobackup3/anirudh/datasets/SDv1-5/gustavosta/res-512/guidance-7-5/images/1_fake_0.5_bicubic"  # Replace with your output folder path
    process_images(input_folder, output_folder, scale_factor=0.5, interpolation=Image.BICUBIC)

