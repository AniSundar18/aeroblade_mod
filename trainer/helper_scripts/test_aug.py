import os
import random
import cv2
import numpy as np
from PIL import Image

def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def histogram_equalization(image):
    if len(image.shape) == 2:  # grayscale image
        return cv2.equalizeHist(image)
    elif len(image.shape) == 3:  # color image
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def subtle_denoising(image, h=3):
    return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            # Randomly choose one of the post-processing techniques
            processing_choice = random.choice(['denoising'])

            if processing_choice == 'gamma':
                processed_image = adjust_gamma(image, gamma=1.05)
            elif processing_choice == 'histogram_equalization':
                processed_image = histogram_equalization(image)
            elif processing_choice == 'denoising':
                processed_image = subtle_denoising(image)

            # Save the processed image
            final_image_path = os.path.join(output_folder, filename)
            cv2.imwrite(final_image_path, processed_image)

if __name__ == "__main__":
    input_folder = "/nobackup3/anirudh/datasets/SDv1-5/cifar-100/res-512/guidance-7-5/images/1_fake"  # Replace with your input folder path
    output_folder = "/nobackup3/anirudh/datasets/SDv1-5/cifar-100/res-512/guidance-7-5/images/1_fake_denoise"  # Replace with your output folder path
    process_images(input_folder, output_folder)

