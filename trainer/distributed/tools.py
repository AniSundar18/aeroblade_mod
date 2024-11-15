from transformers import AutoModel
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from torch.utils.data import DataLoader
import sys
sys.path.append('/nobackup/anirudh/edit_repo/aeroblade_mod/trainer/distributed')
sys.path.append('/nobackup/anirudh/edit_repo/aeroblade_mod/trainer')
from models.networks import TransformerBlock, MLP, MLP_Classifier, old_MLP_Classifier, CNNHead, SelfAttentionBinaryClassification
from utils import get_vae, get_clip, reconstruct, RandomAugment, get_lpips
from diffusers import AutoencoderKL
def cleanup():
    dist.destroy_process_group()

def get_wrapped_models_dino(rank, opt, isVal=False):
    dino = AutoModel.from_pretrained('facebook/dinov2-large', patch_size = 14).to(rank)  
    if opt.use_attn:
        projection_layer = TransformerBlock(num_heads=8, embed_dim=opt.hidden_dim).to(rank)
    elif opt.use_mlp:
        projection_layer = MLP(opt.in_dim, opt.hidden_dim).to(rank)
    elif opt.use_mlp_cls:
        projection_layer = MLP_Classifier(input_dim = opt.in_dim, num_layers = opt.num_layers, use_distance=opt.use_distance).to(rank)
        #projection_layer = old_MLP_Classifier(input_dim = opt.in_dim).to(rank)
    else:
        projection_layer = nn.Linear(opt.in_dim, opt.out_dim).to(rank)
    if not isVal:
        dino = DDP(dino, device_ids=[rank], output_device=rank)
        projection_layer = DDP(projection_layer, device_ids=[rank], output_device=rank)
    return dino, projection_layer



def get_wrapped_models(rank, opt, isVal=False):
    if 'lpips' in opt.distance_metrics:
        _,net, layer = opt.distance_metrics.split("_")
    elif 'CLIP' in opt.distance_metrics:
        net, layer = opt.distance_metrics.split("_")
    ae = get_vae(repo_id=opt.repo_ids[0]).to(rank) #Load the stable diffusion autoencoder
    if opt.vae_path is not None:
        print(f'Loading weights from ', opt.vae_path)
        ae = AutoencoderKL.from_pretrained(opt.vae_path).to(rank)
    if 'CLIP' in net:
        encoder =  get_clip(net=net, layer=layer, spatial=opt.spatial, batch_size=opt.batch_size, num_workers=opt.num_workers, rank=rank)  
    elif 'vgg' in net:
        encoder = get_lpips(net=net, layer=layer, spatial=opt.spatial, batch_size=opt.batch_size, num_workers=opt.num_workers, rank=rank)
    if opt.use_attn:
        projection_layer = SelfAttentionBinaryClassification(input_dim=opt.in_dim, hidden_dim=768, num_heads = 8, output_dim=1).to(rank)
        #projection_layer = TransformerBlock(num_heads=8, embed_dim=opt.hidden_dim).to(rank)
    elif opt.use_mlp:
        projection_layer = MLP(opt.in_dim, opt.hidden_dim).to(rank)
    elif opt.use_mlp_cls:
        projection_layer = MLP_Classifier(input_dim = opt.in_dim, num_layers = opt.num_layers, use_distance=opt.use_distance).to(rank)
        #projection_layer = old_MLP_Classifier(input_dim = opt.in_dim).to(rank)
    elif opt.use_cnn:
        projection_layer = CNNHead(in_channels = opt.in_dim).to(rank)
    else:
        projection_layer = nn.Linear(opt.in_dim, opt.out_dim).to(rank)
    if not isVal:
        ae = DDP(ae, device_ids=[rank], output_device=rank)
        encoder.model = DDP(encoder.model, device_ids=[rank], output_device=rank)
        projection_layer = DDP(projection_layer, device_ids=[rank], output_device=rank)
    return ae, encoder, projection_layer

def prepare(dataset, rank, world_size, batch_size=32, pin_memory=False, num_workers=0, isVal=False):
    if rank is None:
        sampler = None
    else:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, sampler=sampler)
    
    return dataloader


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


