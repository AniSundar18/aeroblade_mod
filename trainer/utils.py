import os
import random
os.chdir('/nobackup3/anirudh/aeroblade/src')

from aeroblade.distances import LPIPS,CLIP
os.chdir('/nobackup3/anirudh/aeroblade/trainer')
from augmenter import Augmenter
from PIL import Image
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)
import torchvision.transforms as transforms

def resize_batch(images, scales=[0.5, 1]):
    # Randomly select a scale from the provided scales
    scale = random.choice(scales)
    
    # Get the dimensions of the first image in the batch
    _, original_height, original_width = images[0].shape
    
    # Calculate the new dimensions based on the selected scale
    new_height = int(original_height * scale)
    new_width = int(original_width * scale)
    
    # Define the resize transform
    resize_transform = transforms.Resize((new_height, new_width))
    
    # Apply the resize transform to all images in the batch
    resized_images = torch.stack([resize_transform(image) for image in images])
    
    return resized_images

def save_network(model, opt, layer, epoch, isBest=False):
    if isBest:
        model_name = f"mlp_{str(layer)}_epoch_best.pth"
    else:
        model_name = f"mlp_{str(layer)}_epoch_{str(epoch)}.pth"
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir) 
    checkpoint_path = os.path.join(opt.save_dir, model_name)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")


def reconstruct(x, ae, seed):
    decode_dtype = next(iter(ae.module.post_quant_conv.parameters())).dtype
    generator = torch.Generator().manual_seed(seed)
    x = x.to(dtype=ae.module.dtype) * 2.0 - 1.0
    latents = retrieve_latents(ae.module.encode(x), generator=generator)
    reconstructions = ae.module.decode(
                        latents.to(decode_dtype), return_dict=False
                    )[0]
    reconstructions = (reconstructions / 2 + 0.5).clamp(0, 1)
    return reconstructions

def reconstruct_simple(x, ae, seed):
    decode_dtype = next(iter(ae.post_quant_conv.parameters())).dtype
    generator = torch.Generator().manual_seed(seed)
    x = x.to(dtype=ae.dtype) * 2.0 - 1.0
    latents = retrieve_latents(ae.encode(x), generator=generator)
    reconstructions = ae.decode(
                        latents.to(decode_dtype), return_dict=False
                    )[0]
    reconstructions = (reconstructions / 2 + 0.5).clamp(0, 1)
    return reconstructions


def get_lpips(net, layer, spatial, batch_size, num_workers, rank=None):
    print('RANK:', rank)
    lpips = LPIPS(
            net=net,
            layer=int(layer),
            spatial=spatial,
            batch_size=batch_size,
            num_workers=num_workers,
            get_diff=True,
            rank=rank
        )
    return lpips


def get_clip(net, layer, spatial, batch_size, num_workers, rank=None):
    clip = CLIP(
            net=net,
            layer=int(layer),
            spatial=spatial,
            batch_size=batch_size,
            num_workers=num_workers,
            get_diff=True,
            rank=rank
        )
    return clip


def get_vae(repo_id):
    pipe = AutoPipelineForImage2Image.from_pretrained(
            repo_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16" if "kandinsky-2" not in repo_id else None,
        )
    return pipe.vae


class RandomAugment:
    def __init__(self, sigma_range=[0,1,2], jpg_qual=[50,60,70,80,90,100], scale_range = [0.5,2], noise_range=[0.01, 0.02], no_aug=False,use_nested=False, final_rez=512):
        self.augmenter = Augmenter(sigma_range=sigma_range, jpg_qual=jpg_qual, scale_range = scale_range, noise_range=noise_range, no_aug=no_aug, use_nested=use_nested, final_rez=final_rez)

    def process_image(self, image: Image.Image) -> Image.Image:
        return self.augmenter.augment(image)

    def __call__(self, image: Image.Image) -> Image.Image:
        return self.process_image(image)
