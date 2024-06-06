"""
Study the effects of resizing.
"""

import argparse
from pathlib import Path

import pandas as pd
from aeroblade.high_level_funcs import compute_rz_distances
from aeroblade.image import extract_patches
from aeroblade.misc import safe_mkdir, write_config


def main(args):
    output_dir = Path("output/04") / args.experiment_id
    safe_mkdir(output_dir)
    write_config(vars(args), output_dir)
    # compute distances, eventually load precomputed distances for real images
    dirs = [args.real_dir] + args.fake_dirs
    distances = compute_rz_distances(
                dirs = dirs,
                size= args.size,
                transforms= args.transforms,
                rz_ratio= args.resize_ratio,
                down_style=args.downsize_style,
                up_style= args.upsize_style,
                repo_ids=args.repo_ids,
                distance_metrics=args.distance_metrics,
                amount=args.amount,
                reconstruction_root=args.reconstruction_root,
                seed=args.seed,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                vae_path = args.vae_path,
                do_over = args.do_over,
                optimize = args.optimize,
                post_transform = args.post_transform
                )


    # store distances
    categoricals = [
        "dir",
        "image_size",
        "repo_id",
        "transform",
        "distance_metric",
        "file",
    ]
    distances[categoricals] = distances[categoricals].astype("category")
    print(distances)
    distances.to_parquet(output_dir / "distances.parquet")

    

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment-id", default="default")

    # images
    parser.add_argument("--real-dir", type=Path, default="data/raw/real")
    parser.add_argument(
        "--fake-dirs",
        type=Path,
        nargs="+",
        default=[
            Path("data/raw/generated/CompVis-stable-diffusion-v1-1-ViT-L-14-openai"),
            Path("data/raw/generated/runwayml-stable-diffusion-v1-5-ViT-L-14-openai"),
            Path(
                "data/raw/generated/stabilityai-stable-diffusion-2-1-base-ViT-H-14-laion2b_s32b_b79k"
            ),
            Path(
                "data/raw/generated/kandinsky-community-kandinsky-2-1-ViT-L-14-openai"
            ),
            Path("data/raw/generated/midjourney-v4"),
            Path("data/raw/generated/midjourney-v5"),
            Path("data/raw/generated/midjourney-v5-1"),
        ],
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
            "lpips_vgg_2",
        ],
    )

    # Resizing stats
    parser.add_argument("--resize_ratio", type=float, default=0.75, help="Resize the image to what percent of the original")
    parser.add_argument("--downsize_style", type=str, default='bilinear', help="style of downsizing")
    parser.add_argument("--upsize_style", type=str, default='bilinear', help="style of upsizing")
    parser.add_argument(
    "--vae_path", type=str, default=None
    )
    parser.add_argument("--size", type=int, default=512, help='Size at which we want the images to be')

    # technical
    parser.add_argument(
        "--reconstruction-root", type=Path, default="data/reconstructions"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)

    parser.add_argument('--do_over', action='store_true', help='Compute reconstructions from scratch')
    parser.add_argument('--optimize', action='store_true', help='Optimize to find a better reconstruction')
    parser.add_argument('--post_transform', action='store_true', help='Apply the same operation on the generated image as well')
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
