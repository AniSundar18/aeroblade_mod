"""
Compute reconstruction distances and detection/attribution results.
"""
import torch
import os
import argparse
from pathlib import Path
from aeroblade.data import ImageFolder
from aeroblade.high_level_funcs import compute_distances
import pandas as pd
from aeroblade.evaluation import tpr_at_max_fpr
from aeroblade.high_level_funcs import compute_distances
from aeroblade.misc import safe_mkdir, write_config
from sklearn.metrics import average_precision_score
from aeroblade.distances import distance_from_config

def main(args):
    output_dir = Path("output/06") / args.experiment_id
    safe_mkdir(output_dir)
    write_config(vars(args), output_dir)
    dist_metric = "lpips_vgg_-1"
    #dist_metric = "CLIP:ViT-L/14_10" 
    layer = 'lpips_vgg_2'
    aps = []
    tprs = []
    #Go through all iterations of reconstruction and calculate tpr@5fpr
    for iteration in range(1,args.iterations):
        real_recon_path = Path(os.path.join(args.real_recon_dir, str(iteration)))
        fake_recon_path = Path(os.path.join(args.fake_recon_dirs, str(iteration)))
        ds_real = ImageFolder(Path(args.real_dir), amount=args.amount)
        ds_fake = ImageFolder(Path(args.fake_dirs), amount=args.amount)
        ds_real_recon = ImageFolder(real_recon_path, amount=args.amount)

        real_dist, _ = distance_from_config(
                        dist_metric,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                    ).compute(
                        ds_a=ds_real,
                        ds_b=ds_real_recon,
                    )
        ds_fake_recon = ImageFolder(fake_recon_path, amount=args.amount)

        fake_dist, _ = distance_from_config(
                        dist_metric,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                    ).compute(
                        ds_a=ds_fake,
                        ds_b=ds_fake_recon,
                    )
        y_score_real = torch.flatten(real_dist[layer])
        y_score_fake = torch.flatten(fake_dist[layer])
        #y_score_real = torch.flatten(real_dist[dist_metric])
        #y_score_fake = torch.flatten(fake_dist[dist_metric])
        #print('REAL:', -1*torch.mean(y_score_real), torch.var(y_score_real))
        #print('FAKE:', -1*torch.mean(y_score_fake), torch.var(y_score_fake))
        y_score = y_score_real.tolist() + y_score_fake.tolist()
        y_true = [0] * len(y_score_real) + [1] * len(y_score_fake)
        ap = average_precision_score(y_true=y_true, y_score=y_score)
        tpr5fpr = tpr_at_max_fpr(y_true=y_true, y_score=y_score, max_fpr=0.05)
        aps.append(ap)
        tprs.append(tpr5fpr)
    print('AP:', aps)
    print('tpr@5fpr:', tprs)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", default="default")

    # images
    parser.add_argument("--real-dir", type=Path, default="data/raw/real")
    parser.add_argument("--real-recon-dir", type=Path, default="data/raw/real")
    parser.add_argument(
        "--fake-dirs",
        type=Path,
        )
    parser.add_argument(
        "--fake-recon-dirs",
        type=Path
        )
    parser.add_argument("--amount", type=int)
    parser.add_argument("--transforms", nargs="*", default=["clean"])
    # autoencoder
    parser.add_argument(
        "--repo-ids",
        nargs="+",
        default=[
            "CompVis/stable-diffusion-v1-1",
            "stabilityai/stable-diffusion-2-base",
            "kandinsky-community/kandinsky-2-1",
        ],
    )

    # distance
    parser.add_argument(
        "--distance-metrics",
        nargs="+",
        default=[
            "lpips_vgg_-1",
        ],
    )

    # technical
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
