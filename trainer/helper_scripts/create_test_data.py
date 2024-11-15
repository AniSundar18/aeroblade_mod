import os
import random
from PIL import Image

def create_directories(base_dir):
    os.makedirs(os.path.join(base_dir, 'og_samples'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'webp_samples'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'jpg_samples'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'resized_256'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'resized_512'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'resized_1024'), exist_ok=True)

def sample_images(image_folder, num_samples):
    all_images = []
    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff')):
                all_images.append(os.path.join(root, file))
    return random.sample(all_images, min(num_samples, len(all_images)))


def process_images(base_dir, image_folder, sampled_images):
    for image_name in sampled_images:
        iname = image_name.split('/')[-1]
        img_path = os.path.join(image_folder, image_name)
        img = Image.open(img_path)

        # Save original sample
        img.save(os.path.join(base_dir, 'og_samples', iname))
        print(os.path.join(base_dir, 'og_samples', iname))
        # Save WebP compressed image
        img.save(os.path.join(base_dir, 'webp_samples', os.path.splitext(iname)[0] + '.webp'), 'WEBP')

        # Save JPEG compressed image with random compression factor
        quality = random.randint(50, 100)
        img.save(os.path.join(base_dir, 'jpg_samples', os.path.splitext(iname)[0] + '.jpg'), 'JPEG', quality=quality)

        # Resize to 256 and save
        img_resized_256 = img.resize((256, 256))
        img_resized_256.save(os.path.join(base_dir, 'resized_256', iname))

        # Resize to 512 and save
        img_resized_512 = img.resize((512, 512))
        img_resized_512.save(os.path.join(base_dir, 'resized_512', iname))

        # Resize to 1024 and save
        img_resized_1024 = img.resize((1024, 1024))
        img_resized_1024.save(os.path.join(base_dir, 'resized_1024', iname))

def main():
    # User inputs
    image_folder = "/nobackup3/anirudh/datasets/coco/unlabeled2017"
    test_directory = "/nobackup3/anirudh/datasets/benchmark/real/coco"
    num_samples = 500
    base_dir = test_directory
    #base_dir = os.path.join(os.getcwd(), test_directory)
    create_directories(base_dir)

    sampled_images = sample_images(image_folder, num_samples)
    process_images(base_dir=base_dir, image_folder = image_folder, sampled_images = sampled_images)

    print(f"Processing complete. All images saved under the directory: {base_dir}")

if __name__ == "__main__":
    main()




