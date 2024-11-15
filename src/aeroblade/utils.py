from PIL import Image
from aeroblade.augmenter import Augmenter
class RandomAugment:
    def __init__(self, sigma_range=[0,1,2], jpg_qual=[50,60,70,80,90,100], scale_range = [0.5,2], noise_range=[0.01, 0.02]):
        self.augmenter = Augmenter(sigma_range=sigma_range, jpg_qual=jpg_qual, scale_range = scale_range, noise_range=noise_range)

    def process_image(self, image: Image.Image) -> Image.Image:
        return self.augmenter.augment(image)

    def __call__(self, image: Image.Image) -> Image.Image:
        return self.process_image(image)
