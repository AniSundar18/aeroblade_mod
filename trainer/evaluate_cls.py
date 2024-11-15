import os
from PIL import Image
import torch.distributed as dist
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from sklearn.metrics import average_precision_score
os.chdir('/nobackup3/anirudh/aeroblade/src')
from aeroblade.evaluation import tpr_at_max_fpr
from aeroblade.models import get_model
os.chdir('/nobackup3/anirudh/aeroblade/trainer')
from dataset import ImageFolder
from utils import get_vae, get_clip, reconstruct_simple, RandomAugment, save_network
from distributed.tools import setup, cleanup, prepare, get_wrapped_models
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def statistics(data):
    if not data:
        print("List is empty")
        return

    n = len(data)
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n
    maximum = max(data)
    minimum = min(data)

    print(f"Mean: {mean}")
    print(f"Variance: {variance}")
    print(f"Max: {maximum}")
    print(f"Min: {minimum}")



def validate(rank, opt, val_loaders, ae, clip, projection_layer, epoch):
    with torch.no_grad():
        projection_layer.eval() #Set to eval mode, Dropout is present in model
        for idx, val_loader in enumerate(val_loaders):
            if idx == 1:
                break
                print('-----------Normal-------------')
            
            real_loader = val_loader[0]
            fake_loader = val_loader[1]
            y_score_real = []
            y_score_fake = []
            for step, (x_real,_) in tqdm(enumerate(real_loader)):
                diff = clip.get_diff(x_real.to(rank), reconstruct_simple(x_real.to(rank), ae, seed=opt.seed), use_cat=opt.use_cat, rez=opt.pre_rez) #Compute the difference vector
                if opt.use_distance:
                    norms = torch.norm(diff, dim=1, keepdim=True)
                    diff = torch.cat((diff, norms), dim=1)
                preds = projection_layer(diff)
                y_score_real.append(preds)
            y_score_real = torch.flatten(torch.stack(y_score_real))
            #CHANGE WHENEVER BACK TO (x_fake, _)
            for step, (x_fake,_) in tqdm(enumerate(fake_loader)):
                diff = clip.get_diff(x_fake.to(rank), reconstruct_simple(x_fake.to(rank), ae, seed=opt.seed), use_cat=opt.use_cat, rez=opt.pre_rez) #Compute the difference vector
                if opt.use_distance:
                    norms = torch.norm(diff, dim=1, keepdim=True)
                    diff = torch.cat((diff, norms), dim=1)
                preds = projection_layer(diff)
                y_score_fake.append(preds)
            y_score_fake = torch.flatten(torch.stack(y_score_fake))
            print('Real')
            statistics(data=y_score_real.tolist())
            print('--------------------------------------------------------------')
            print('Fake')
            statistics(data=y_score_fake.tolist())
            print('--------------------------------------------------------------')
            y_score = y_score_real.tolist() + y_score_fake.tolist()
            y_true = [0] * len(y_score_real) + [1] * len(y_score_fake)
            ap = average_precision_score(y_true=y_true, y_score=y_score)
            tpr5fpr = tpr_at_max_fpr(y_true=y_true, y_score=y_score, max_fpr=0.05)
            return ap, tpr5fpr
            #print('Validation Statistics at {str(epoch)}, process {str(rank)}')
            #print('Validation AP: ', ap)
            #print('Validation tpr@5fpr: ', tpr5fpr)
    

def evaluate(opt, val_loaders=None):
    _, layer = opt.distance_metrics.split("_")
    ae, clip, projection_layer = get_wrapped_models(rank='cuda', opt=opt, isVal=True)
    print(projection_layer)
    if opt.use_mlp_cls or opt.use_attn:
        projection_layer.load_state_dict(torch.load(opt.checkpoint)) 
    ap, tpr5fpr = validate('cuda', opt, val_loaders, ae, clip, projection_layer, epoch=0)
    print('Validation AP: ', ap)
    print('Validation tpr@5fpr: ', tpr5fpr)

def main():
    opt = parse_args()
    
    args_dict = vars(opt)
    print(args_dict)
    scale = 0.75
    size = 512

    random_transform = transforms.Compose(
                                    [transforms.Resize((256,256), interpolation=Image.BICUBIC),
                                    #transforms.Resize(256, interpolation=Image.BICUBIC),
                                    transforms.ToTensor()])#
    #random_transform = RandomAugment()
    print(random_transform)
    ds_real = ImageFolder(paths=opt.real_dir, amount=opt.amount, transform=random_transform)
    ds_fake = ImageFolder(paths=opt.fake_dirs, amount=opt.amount, transform=random_transform)
    data_loaders = ([prepare(dataset=ds_real, rank=None, world_size=opt.world_size, batch_size=opt.batch_size, pin_memory=False, num_workers=0, isVal=True), prepare(dataset=ds_fake, rank=None, world_size=opt.world_size, batch_size=opt.batch_size, pin_memory=False, num_workers=0, isVal=True)], [])
    #data_loader = prepare(dataset=ds, rank=None, world_size=opt.world_size, batch_size=opt.batch_size, pin_memory=False, num_workers=0)
    evaluate(opt = opt, val_loaders = data_loaders)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", type=Path, default=Path("/nobackup3/anirudh/aeroblade/data/inference/real_val"))
    parser.add_argument(
        "--fake-dirs",
        type=Path,
        default=Path("/nobackup3/anirudh/aeroblade/data/inference/gen_val"),
        )

    parser.add_argument("--amount", type=int)
    parser.add_argument(
        "--repo-ids",
        nargs="+",
        default=[
            "CompVis/stable-diffusion-v1-1",
            "stabilityai/stable-diffusion-2-base",
            "kandinsky-community/kandinsky-2-1",
        ],
        )
    parser.add_argument(
        "--checkpoint", type=Path, default="checkpoints"
    )
    parser.add_argument(
        "--distance-metrics",
        default="lpips_vgg_-1",
    )
        # technical
    parser.add_argument(
    "--vae_path", type=str, default=None
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument('--spatial', action='store_true', help='Use all patches for CLIP distance')
    parser.add_argument("--out_dim", type=int, default=1024)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--in_dim", type=int, default=1024)
    parser.add_argument('--use_mlp', action='store_true', help='Use MLP') 
    parser.add_argument('--use_mlp_cls', action='store_true', help='Use MLP Classifier')
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--pre_rez", type=int, default=512)
    parser.add_argument('--use_attn', action='store_true', help='Use Attention')
    parser.add_argument('--use_cnn', action='store_true', help='Use CNN head')
    parser.add_argument('--use_distance', action='store_true', help='Use Distance')
    parser.add_argument('--use_nested', action='store_true', help='Use nested augmentations to train')
    parser.add_argument('--use_cat', action='store_true', help='Concatenate the representations instead of taking difference')
    #Distributed settings
    parser.add_argument("--world-size", type=int, default=4)

    return parser.parse_args()
    

if __name__ == '__main__':
    main()
