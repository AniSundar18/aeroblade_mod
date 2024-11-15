from scipy.ndimage.filters import gaussian_filter
import torchvision
import torch
import torchvision.datasets as datasets
import torchvision.transforms.v2 as tf
import torchvision.transforms.functional as TF
from random import random, choice
from io import BytesIO
from PIL import Image, ImageFilter
from PIL import ImageFile
import cv2
import numpy as np



def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]

def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img

jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, method):
    return Image.fromarray(method(img, compress_val))

def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")

def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)

class Augmenter:
    def __init__(self, sigma_range=[0,1,2], jpg_qual=[50,60,70,80,90,100], scale_range = [0.5,2], noise_range=[0.01, 0.02]):
        self.sigma_range = sigma_range
        self.jpg_qual = jpg_qual
        self.jpg_method = [cv2_jpg, pil_jpg]
        self.rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}
        self.process = {'blur': self.blur,
                'compress':self.compress,
                'resize':self.resize,
                'no_op':None
                }
        self.rz_keys = list(self.rz_dict.keys())
        self.scale_range = scale_range
        self.noise_range = noise_range

    def blur(self, img):
        #img = np.array(img)
        sigma = sample_discrete(s = self.sigma_range)
        img = img.filter(ImageFilter.GaussianBlur(sigma))
        #gaussian_blur(img, sigma)
        return img


    def compress(self, img):
        img = np.array(img)
        method = sample_discrete(self.jpg_method)
        qual = sample_discrete(self.jpg_qual)
        img = jpeg_from_key(img, qual, method)
        return img
        

    def resize(self, img):
        width, height = img.size
        scale = sample_continuous(self.scale_range)
        uidx = sample_discrete([0,1,2,3])
        up_style = self.rz_dict[self.rz_keys[uidx]]
        didx = sample_discrete([0,1,2,3])
        down_style = self.rz_dict[self.rz_keys[didx]]
        resize = tf.Compose([
                        tf.Resize((int(scale*height), int(scale*width)), interpolation=up_style),
                        tf.Resize((height, width), interpolation=down_style)
                        ])
        return resize(img)
        

    def add_noise(self, img):
        noise = sample_continuous(self.noise_range)
        toTensor = torchvision.transforms.ToTensor()
        img = toTensor(img)
        img = img + noise * torch.randn_like(img)
        toPIL = tf.ToPILImage()
        return toPIL(img)
    
    def augment(self, img):
        toTensor = torchvision.transforms.ToTensor()
        operations = list(self.process.keys())
        cidx = sample_discrete([0,1,2,3])
        if operations[cidx] != 'no_op':
            print(operations[cidx])
            #post_proc = self.process['resize'](img)
            post_proc = self.process[operations[cidx]](img)
            return toTensor(post_proc)
        else:
            return toTensor(img)

    

