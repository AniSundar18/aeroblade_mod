import os
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
from utils import get_vae, get_clip, reconstruct, RandomAugment, save_network
from distributed.tools import setup, cleanup, prepare, get_wrapped_models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validate(rank, opt, val_loaders, ae, clip, projection_layer, epoch):
    with torch.no_grad():
        projection_layer.eval()
        for idx, val_loader in enumerate(val_loaders):
            if idx == 1:
                break
                print('-----------Normal-------------')
            
            real_loader = val_loader[0]
            fake_loader = val_loader[1]
            real_loader.sampler.set_epoch(0)
            fake_loader.sampler.set_epoch(0)
            y_score_real = []
            y_score_fake = []
            for step, (x_real, _) in enumerate(real_loader):
                diff = clip.get_diff(x_real.to(rank), reconstruct(x_real.to(rank), ae, seed=opt.seed)) #Compute the difference vector
                projs = projection_layer(diff)
                if opt.spatial:
                    dists = torch.norm(projs, dim=-1, keepdim=True)
                    dists = torch.mean(dists, dim=0)
                else:
                    dists= torch.norm(projs, dim=1)
                y_score_real.append(dists)
            y_score_real = -torch.flatten(torch.stack(y_score_real))
            for step, (x_fake, _) in enumerate(fake_loader):
                diff = clip.get_diff(x_fake.to(rank), reconstruct(x_fake.to(rank), ae, seed=opt.seed)) #Compute the difference vector
                projs = projection_layer(diff)
                if opt.spatial:
                    dists = torch.norm(projs, dim=-1, keepdim=True)
                    dists = torch.mean(dists, dim=0)
                else:
                    dists= torch.norm(projs, dim=1)
                y_score_fake.append(dists)
            y_score_fake = -torch.flatten(torch.stack(y_score_fake))
            y_score = y_score_real.tolist() + y_score_fake.tolist()
            y_true = [0] * len(y_score_real) + [1] * len(y_score_fake)
            ap = average_precision_score(y_true=y_true, y_score=y_score)
            tpr5fpr = tpr_at_max_fpr(y_true=y_true, y_score=y_score, max_fpr=0.05)
            return ap, tpr5fpr
            #print('Validation Statistics at {str(epoch)}, process {str(rank)}')
            #print('Validation AP: ', ap)
            #print('Validation tpr@5fpr: ', tpr5fpr)
    

def train(rank, opt, train_loader, val_loaders=None):
    _, layer = opt.distance_metrics.split("_")
    l2_loss = nn.MSELoss(reduction='none')
    ae, clip, projection_layer = get_wrapped_models(rank=rank, opt=opt)
    optimizer = torch.optim.AdamW(projection_layer.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
    print(projection_layer)
    with torch.no_grad():
        if not (opt.use_mlp or opt.use_attn):
            projection_layer.module.weight.copy_(torch.eye(opt.in_dim, opt.out_dim))
            projection_layer.module.bias.zero_()
    if opt.ema:
        moving_avg_weights = projection_layer.module.weight.data.clone().to(rank)
        moving_avg_bias = projection_layer.module.bias.data.clone().to(rank)
    if rank == 0:
        best_ap, best_tpr5fpr = validate(rank, opt, val_loaders, ae, clip, projection_layer, epoch = -1) 
        print('AP: ', best_ap)
        print('TPR5FPR: ', best_tpr5fpr)
    dist.barrier()
    for eidx in range(opt.epochs):
        train_loader.sampler.set_epoch(eidx)
        for step, (x, x_aug) in enumerate(train_loader):
            with torch.no_grad():
                diff_og = clip.get_diff(x.to(rank), reconstruct(x.to(rank), ae, seed=opt.seed)) #Compute the difference vector
                diff_aug = clip.get_diff(x_aug.to(rank), reconstruct(x_aug.to(rank), ae, seed=opt.seed)) #Compute the difference vector
            proj_diff = projection_layer(diff_aug)
            #print(proj_diff.shape, diff_og.shape)
            #loss = l2_loss(proj_diff, diff_og)
            if opt.spatial:
                dists = torch.norm(proj_diff - diff_og, dim=-1, keepdim=True)**2
                loss = torch.flatten(torch.mean(dists, dim=0))
                loss = torch.mean(loss)
            else:
                loss = torch.mean(torch.norm(proj_diff - diff_og, dim=1)**2)
            #print(f"Epoch {eidx}, step {step}: {loss}")
            #print('Norm:', torch.norm(diff_og[0]), torch.norm(proj_diff[0]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if opt.ema:
                projection_layer.module.weight.data = opt.alpha * moving_avg_weights + (1 - opt.alpha) * projection_layer.module.weight.data
                projection_layer.module.bias.data = opt.alpha * moving_avg_bias + (1 - opt.alpha) * projection_layer.module.bias.data
            if step%10 == 0:
                print(f"Epoch {eidx}/{step}: {loss}")
                if rank == 0:
                    ap, tpr5fpr = validate(rank, opt, val_loaders, ae, clip, projection_layer, epoch = -1)
                    print('AP: ' ,ap)
                    print('TPR5FPR: ' ,tpr5fpr)
                dist.barrier()
        #validate(rank, opt, val_loaders, ae, clip, projection_layer, epoch=eidx)
        if rank == 0 and eidx % opt.save_freq == 0:
            if isinstance(projection_layer, nn.parallel.DistributedDataParallel):
                if ap > best_ap and tpr5fpr > best_tpr5fpr:
                    save_network(model = projection_layer.module, opt=opt, layer=layer, epoch=eidx, isBest=True)
                    best_ap = ap
                    best_tpr5fpr = tpr5fpr
                save_network(model = projection_layer.module, opt=opt, layer=layer, epoch=eidx, isBest=False)
        #Loss is difference between the two vectors
    cleanup()

def main(rank, world_size):
    opt = parse_args()
    
    args_dict = vars(opt)
    print(args_dict)

    # setup the process groups
    setup(rank, world_size)

    random_transform = RandomAugment(no_aug=opt.no_aug)
    ds = ImageFolder(paths=[opt.real_dir, opt.fake_dirs], amount=opt.amount, transform=random_transform)
    data_loader = prepare(dataset=ds, rank=rank, world_size=opt.world_size, batch_size=opt.batch_size, pin_memory=False, num_workers=0)
    if rank == 0:
        vs = [ImageFolder(paths=opt.real_val, amount=opt.val_amount), ImageFolder(paths=opt.fake_val, amount=opt.val_amount)]
        vs_normal = [ImageFolder(paths=opt.real_val_normal, amount=opt.val_amount), ImageFolder(paths=opt.fake_val_normal, amount=opt.val_amount)]
        val_loaders = ([prepare(dataset=vs[0], rank=rank, world_size=opt.world_size, batch_size=opt.batch_size, pin_memory=False, num_workers=0, isVal=True), prepare(dataset=vs[1], rank=rank, world_size=opt.world_size, batch_size=opt.batch_size, pin_memory=False, num_workers=0, isVal=True)], [prepare(dataset=vs_normal[0], rank=rank, world_size=opt.world_size, batch_size=opt.batch_size, pin_memory=False, num_workers=0, isVal=True), prepare(dataset=vs_normal[1], rank=rank, world_size=opt.world_size, batch_size=opt.batch_size, pin_memory=False, num_workers=0, isVal=True)]
            ) 
    else:
        val_loaders = [None, None, None, None] #Hack fix later
    train(rank=rank, opt = opt, train_loader = data_loader, val_loaders = val_loaders)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", type=Path, default=Path("/nobackup3/anirudh/datasets/redcaps/samples"))
    parser.add_argument(
        "--fake-dirs",
        type=Path,
        default=Path("/nobackup3/anirudh/datasets/laion_generated_images/1_fake"),
        )
    parser.add_argument("--real-val", type=Path, default=Path("/nobackup3/anirudh/aeroblade/data/inference/real_val"))
    parser.add_argument("--real-val-normal", type=Path, default=Path("/nobackup3/anirudh/aeroblade/data/raw/real/00000"))
    parser.add_argument(
        "--fake-val",
        type=Path,
        default=Path("/nobackup3/anirudh/aeroblade/data/inference/gen_val"),
        )
    parser.add_argument(
        "--fake-val-normal",
        type=Path,
        default=Path("/nobackup3/anirudh/aeroblade/data/raw/generated/runwayml-stable-diffusion-v1-5-ViT-L-14-openai"),
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
        "--save_dir", type=Path, default="checkpoints"
    )
    parser.add_argument("--val_amount", type=int, default=120)
    parser.add_argument(
        "--distance-metrics",
        default="lpips_vgg_-1",
    )
        # technical
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001, help='Learning Rate')
    parser.add_argument("--beta1", type=float, default=0.9, help='beta1 for Adam')
    parser.add_argument("--alpha", type=float, default=0.75, help='weight on moving average')
    parser.add_argument("--weight_decay", type=float, default=0.0, help='Optimizer weight decay')
    parser.add_argument('--spatial', action='store_true', help='Use all patches for CLIP distance')
    parser.add_argument("--out_dim", type=int, default=1024)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--in_dim", type=int, default=1024)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument('--ema', action='store_true', help='Use EMA while updating weights')
    parser.add_argument('--use_mlp', action='store_true', help='Use MLP') 
    parser.add_argument('--use_attn', action='store_true', help='Use Attention')
    parser.add_argument('--no_aug', action='store_true', help='No Augmentation')
    #Distributed settings
    parser.add_argument("--world-size", type=int, default=4)

    return parser.parse_args()
    

if __name__ == '__main__':
    world_size = 3
    mp.spawn(
        main,
        args=[world_size],
        nprocs=world_size
    )
