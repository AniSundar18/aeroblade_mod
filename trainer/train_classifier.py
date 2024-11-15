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
# os.chdir('/nobackup/anirudh/edit_repo/aeroblade_mod/src')
# from aeroblade.evaluation import tpr_at_max_fpr
# from aeroblade.models import get_model
# os.chdir('/nobackup/anirudh/edit_repo/aeroblade_mod/trainer')
import sys
sys.path.append('/nobackup/anirudh/edit_repo/aeroblade_mod/src')
sys.path.append('/nobackup/anirudh/edit_repo/aeroblade_mod/trainer')

from aeroblade.evaluation import tpr_at_max_fpr
from aeroblade.models import get_model
from dataset import ImageFolder
from utils import get_vae, get_clip, reconstruct, RandomAugment, resize_batch, save_network
from distributed.tools import setup, cleanup, prepare, get_wrapped_models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def validate(rank, opt, val_loader, ae, encoder, projection_layer, epoch, tform=None):

    to_pil = transforms.ToPILImage()
    with torch.no_grad():            
        real_loader = val_loader[0]
        fake_loader = val_loader[1]
        real_loader.sampler.set_epoch(0)
        fake_loader.sampler.set_epoch(0)
        y_score_real = []
        y_score_fake = []
        for step, (x_real, _) in enumerate(real_loader):
            if opt.perturb:
                x_pil = [tform(to_pil(j)) for j in x_real]
                x_pil = torch.stack(x_pil)
                diff = encoder.get_diff(x_real.to(rank), x_pil.to(rank), rez=opt.pre_rez)
            else:
                if 'lpips' in opt.distance_metrics:
                    diff = encoder.diff(x_real.to(rank), reconstruct(x_real.to(rank), ae, seed=opt.seed), use_cat=opt.use_cat, rez=opt.pre_rez)
                else:
                    diff = encoder.get_diff(x_real.to(rank), reconstruct(x_real.to(rank), ae, seed=opt.seed), use_cat=opt.use_cat, rez=opt.pre_rez)
            if opt.use_distance:
                norms = torch.norm(diff, dim=1, keepdim=True)
                diff = torch.cat((diff, norms), dim=1)
            preds = projection_layer(diff)
            y_score_real.append(preds)
        y_score_real = torch.flatten(torch.stack(y_score_real))
        for step, (x_fake, _) in enumerate(fake_loader):
            if opt.perturb:
                x_pil = [tform(to_pil(j)) for j in x_fake]
                x_pil = torch.stack(x_pil)
                diff = encoder.get_diff(x_fake.to(rank), x_pil.to(rank), rez=opt.pre_rez)
            else:
                if 'lpips' in opt.distance_metrics:
                    diff = encoder.diff(x_fake.to(rank), reconstruct(x_fake.to(rank), ae, seed=opt.seed), use_cat=opt.use_cat, rez=opt.pre_rez)
                else:
                    diff = encoder.get_diff(x_fake.to(rank), reconstruct(x_fake.to(rank), ae, seed=opt.seed), use_cat=opt.use_cat, rez=opt.pre_rez)
            if opt.use_distance:
                norms = torch.norm(diff, dim=1, keepdim=True)
                diff = torch.cat((diff, norms), dim=1)
            preds = projection_layer(diff)
            y_score_fake.append(preds)
        y_score_fake = torch.flatten(torch.stack(y_score_fake))
        y_score = y_score_real.tolist() + y_score_fake.tolist()
        y_true = [0] * len(y_score_real) + [1] * len(y_score_fake)
        ap = average_precision_score(y_true=y_true, y_score=y_score)
        tpr5fpr = tpr_at_max_fpr(y_true=y_true, y_score=y_score, max_fpr=0.05)
        return ap, tpr5fpr


def train(rank, opt, train_loaders, val_loader=None):
    bce_loss = nn.BCELoss()
    to_pil = transforms.ToPILImage()
    real_loader = train_loaders[0]
    fake_loader = train_loaders[1]
    sigmoid = nn.Sigmoid()
    if 'lpips' in opt.distance_metrics:
        _, net, layer = opt.distance_metrics.split("_")
    elif 'CLIP' in opt.distance_metrics:
        _, layer = opt.distance_metrics.split("_")
    ae, encoder, projection_layer = get_wrapped_models(rank=rank, opt=opt)
    if opt.perturb:
        ae = None
        tform = RandomAugment()
    else:
        tform = None
    optimizer = torch.optim.AdamW(projection_layer.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
    print(projection_layer)
    if rank==0:
        best_ap, best_tpr5fpr = validate(rank, opt, val_loader, ae, encoder, projection_layer, epoch = -1, tform=tform) 
        print('AP: ', best_ap)
        print('TPR5FPR: ', best_tpr5fpr)
    dist.barrier()
    for eidx in range(opt.epochs):
        real_loader.sampler.set_epoch(eidx)
        fake_loader.sampler.set_epoch(eidx)
        for step, ((xr, xr_aug), (xf, xf_aug)) in enumerate(zip(real_loader, fake_loader)):
            x_real = torch.cat((xr, xr_aug))
            x_fake = torch.cat((xf, xf_aug))
            y_real = torch.zeros(x_real.shape[0])
            y_fake = torch.ones(x_fake.shape[0])
            x = torch.cat((x_real, x_fake))
            x = resize_batch(images=x, scales=[0.5, 1, 2])
            y = torch.cat((y_real, y_fake)).to(rank)
            with torch.no_grad():
                if opt.perturb:
                    x_pil = [tform(to_pil(j)) for j in x]
                    x_pil = torch.stack(x_pil)
                    diff = encoder.get_diff(x.to(rank), x_pil.to(rank), use_cat=opt.use_cat, rez=opt.pre_rez)
                else:
                    if 'lpips' in opt.distance_metrics:
                        print('Training rank:', rank)
                        diff = encoder.diff(x.to(rank), reconstruct(x.to(rank), ae, seed=opt.seed), use_cat=opt.use_cat, rez=opt.pre_rez)
                    else:
                        diff = encoder.get_diff(x.to(rank), reconstruct(x.to(rank), ae, seed=opt.seed), use_cat=opt.use_cat, rez=opt.pre_rez) #Compute the difference vector
            if opt.use_distance:
                norms = torch.norm(diff, dim=1, keepdim=True)
                diff = torch.cat((diff, norms), dim=1)
            projs = projection_layer(diff)
            preds = torch.flatten(projs)
            loss = bce_loss(preds, y)
            #print(f"Epoch {eidx}, step {step}: {loss}")
            #print('Norm:', torch.norm(diff_og[0]), torch.norm(proj_diff[0]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step%10 == 0:
                print(f"Epoch {eidx}/{step}: {loss}")
                if rank == 0:
                    ap, tpr5fpr = validate(rank, opt, val_loader, ae, encoder, projection_layer, epoch = -1, tform=tform)
                    print('AP: ' ,ap)
                    print('TPR5FPR: ' ,tpr5fpr)
                dist.barrier()
        #validate(rank, opt, val_loaders, ae, clip, projection_layer, epoch=eidx)
        if rank == 0 and eidx % opt.save_freq == 0:
            if isinstance(projection_layer, nn.parallel.DistributedDataParallel):
                if ap >= best_ap and tpr5fpr >= best_tpr5fpr:
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

    random_transform = RandomAugment(no_aug=opt.no_aug, use_nested=opt.use_nested, final_rez=opt.final_rez)
    ds_real = ImageFolder(paths=opt.real_dir, amount=opt.amount, transform=random_transform, final_rez=opt.final_rez)
    ds_fake = ImageFolder(paths=opt.fake_dirs, amount=opt.amount, transform=random_transform, final_rez=opt.final_rez)
    real_loader = prepare(dataset=ds_real, rank=rank, world_size=opt.world_size, batch_size=opt.batch_size, pin_memory=False, num_workers=0)
    fake_loader = prepare(dataset=ds_fake, rank=rank, world_size=opt.world_size, batch_size=opt.batch_size, pin_memory=False, num_workers=0)
    if rank == 0:
        vs = [ImageFolder(paths=opt.real_val, amount=opt.val_amount, final_rez=opt.final_rez), 
                ImageFolder(paths=opt.fake_val, amount=opt.val_amount, final_rez=opt.final_rez)]

        val_loader =[prepare(dataset=vs[0], rank=rank, world_size=opt.world_size, batch_size=opt.batch_size, pin_memory=False, num_workers=0, isVal=True),
                     prepare(dataset=vs[1], rank=rank, world_size=opt.world_size, batch_size=opt.batch_size, pin_memory=False, num_workers=0, isVal=True)]

             
    else:
        val_loader = None #Hack fix later
    train(rank=rank, opt = opt, train_loaders = [real_loader, fake_loader], val_loader = val_loader)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", type=Path, default=Path("/nobackup3/anirudh/datasets/redcaps/samples"))
    parser.add_argument(
        "--fake-dirs",
        type=Path,
        default=Path("/nobackup3/anirudh/datasets/laion_generated_images/1_fake"),
        )
    parser.add_argument("--real-val", type=Path, default=Path("/nobackup/anirudh/edit_repo/aeroblade/data/inference/real_val"))
    parser.add_argument(
        "--fake-val",
        type=Path,
        default=Path("/nobackup/anirudh/edit_repo/aeroblade/data/inference/gen_val"),
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
    "--vae_path", type=str, default=None
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
    parser.add_argument("--out_dim", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--in_dim", type=int, default=1024)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--final_rez", type=int, default=512)
    parser.add_argument("--pre_rez", type=int, default=512)
    parser.add_argument('--ema', action='store_true', help='Use EMA while updating weights')
    parser.add_argument('--use_mlp_cls', action='store_true', help='Use MLP for CLS')
    parser.add_argument('--use_mlp', action='store_true', help='Use MLP')
    parser.add_argument('--use_attn', action='store_true', help='Use Attention')
    parser.add_argument('--use_cnn', action='store_true', help='Use CNN head')
    parser.add_argument('--use_distance', action='store_true', help='Use Distance also in the representation')
    parser.add_argument('--perturb', action='store_true', help='Use Augmentations instead of autoencoders')
    parser.add_argument('--no_aug', action='store_true', help='No Augmentation')
    parser.add_argument('--use_nested', action='store_true', help='Use nested augmentations to train')
    parser.add_argument('--use_cat', action='store_true', help='concatenate the vectors instead of taking difference')
    #Distributed settings
    parser.add_argument("--world-size", type=int, default=4)

    return parser.parse_args()
    

if __name__ == '__main__':
    world_size = 8
    mp.spawn(
        main,
        args=[world_size],
        nprocs=world_size
    )
