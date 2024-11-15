from pathlib import Path
import os
from typing import Callable, Optional
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as tf
from tqdm import tqdm
from aeroblade.networks import MLP
from aeroblade.utils import RandomAugment
from aeroblade.complexities import complexity_from_config
from aeroblade.data import ImageFolder, read_files
from aeroblade.distances import distance_from_config
from aeroblade.image import compute_reconstructions
from aeroblade.transforms import transform_from_config

def compute_rz_distances(
    dirs: list[Path],
    size: int,
    transforms: list[str | Callable],
    rz_ratio: float,
    down_style: str,
    up_style: str,
    repo_ids: list[str],
    distance_metrics: list[str],
    amount: Optional[int],
    reconstruction_root: Path,
    seed: int,
    batch_size: int,
    num_workers: int,
    compute_max: bool = True,
    vae_path=None,
    do_over = True,
    optimize = False,
    post_transform=False,
    save_spat_dist=None,
    **distance_kwargs,
) -> pd.DataFrame:
    """Compute distances between original and reconstructed images."""
    # set up progress bar
    pbar = tqdm(
        desc="PROGRESS (compute_distances)",
        total=len(transforms) * len(dirs) * len(repo_ids) * len(distance_metrics),
    )
    #amount=1000
    distances = []
    print('T: ', transforms)
    # iterate over transforms
    for transform_config in transforms:
        
        if down_style == 'nearest':
            drz = tf.Resize((int(rz_ratio*size), int(rz_ratio*size)), interpolation=Image.NEAREST)
        elif down_style == 'bilinear':
            drz = tf.Resize((int(rz_ratio*size), int(rz_ratio*size)), interpolation=Image.BILINEAR)
        elif down_style == 'bicubic':
            drz = tf.Resize((int(rz_ratio*size), int(rz_ratio*size)), interpolation=Image.BICUBIC)
        elif down_style == 'lanczos':
            drz = tf.Resize((int(rz_ratio*size), int(rz_ratio*size)), interpolation=Image.LANCZOS)

        if up_style == 'nearest':
            urz = tf.Resize((size, size), interpolation=Image.NEAREST)
        elif up_style == 'bilinear':
            urz = tf.Resize((size, size), interpolation=Image.BILINEAR)
        elif up_style == 'bicubic':
            urz = tf.Resize((size, size), interpolation=Image.BICUBIC)
        elif up_style == 'lanczos':
            urz = tf.Resize((size, size), interpolation=Image.LANCZOS)

        rz = tf.Compose([
                        drz,
                        urz
                    ])
        transform = tf.Compose(
                [
                    rz,
                    tf.ToImage(),
                    tf.ToDtype(torch.float32, scale=True),
                ]
            )
        # iterate over directories
        recon_paths = []
        for idx, dir in enumerate(dirs):
            # first path is resized before reconstruction
            if idx==0:
                ds = ImageFolder(dir, amount=amount, transform=transform)
            elif idx==1:
                ds = ImageFolder(dir, amount=amount)
            print('_______________________________')
            print(dir)
            print('_______________________________')
            # iterate over autoencoder repo_ids
            for repo_id in repo_ids:
                rec_paths = compute_reconstructions(
                    ds,
                    repo_id=repo_id,
                    output_root=reconstruction_root,
                    seed=seed,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    vae_path=vae_path,
                    do_over = do_over,
                    optimize = optimize
                    )
                if post_transform and idx == 1:
                    # The second path reconstructions are resized (post_processing)
                    ds_rec = ImageFolder(rec_paths, transform=transform)
                else:
                    ds_rec = ImageFolder(rec_paths)
                recon_paths.append(ds_rec)
        
        
        # iterate over distance metrics
        ds = recon_paths[0]
        ds_rec = recon_paths[1]
        for dist_metric in distance_metrics:
            dist_dict, files = distance_from_config(
                dist_metric,
                batch_size=batch_size,
                num_workers=num_workers,
                **distance_kwargs,
            ).compute(
                ds_a=ds,
                ds_b=ds_rec,
            )
            if save_spat_dist is not None:
                file_name = dist_metric + '.pth'
                file_path = os.path.join(save_spat_dist, file_name)
                torch.save(dist_dict, file_path)

            for dist_name, dist_tensor in dist_dict.items():
                if not distance_kwargs.get("spatial", False):
                    dist_tensor = dist_tensor.squeeze(1, 2, 3)
                df = pd.DataFrame(
                    {
                        "dir": str(dir),
                        "image_size": int(ds[0][0].shape[-1]),
                        "repo_id": repo_id,
                        "transform": transform_config,
                        "distance_metric": dist_name,
                        "file": files,
                        "distance": list(dist_tensor.numpy()),
                    }
                )
                distances.append(df)
            pbar.update()

    distances = pd.concat(distances)

    # determine maximum distance over all repo_ids for each file
    if compute_max:
        maxima = []
        for group_keys, group_df in distances.groupby(
            group_cols := ["dir", "image_size", "transform", "distance_metric"],
            sort=False,
        ):
            max_values = group_df.groupby("file").apply(
                lambda df: np.stack(df.distance).max(axis=0)
            )
            max_df = {col: key for col, key in zip(group_cols, group_keys)}
            max_df.update(
                {
                    "repo_id": "max",
                    "file": max_values.index.values,
                    "distance": max_values.values,
                }
            )
            maxima.append(pd.DataFrame(max_df))

        distances = pd.concat([distances, *maxima]).sort_values("dir", kind="stable")
    distances = distances.reset_index(drop=True)
    return distances


def compute_distances(
    dirs: list[Path],
    transforms: list[str | Callable],
    repo_ids: list[str],
    distance_metrics: list[str],
    amount: Optional[int],
    reconstruction_root: Path,
    seed: int,
    batch_size: int,
    num_workers: int,
    compute_max: bool = True,
    spatial: bool = False,
    vae_path=None,
    do_over = True,
    optimize = False,
    scale=1.0,
    post_transform=False,
    save_spat_dist=None,
    iterations=1,
    checkpoint=None,
    **distance_kwargs,
) -> pd.DataFrame:
    """Compute distances between original and reconstructed images."""
    # set up progress bar
    pbar = tqdm(
        desc="PROGRESS (compute_distances)",
        total=len(transforms) * len(dirs) * len(repo_ids) * len(distance_metrics),
    )
    #amount=1000
    if checkpoint is not None:
        projection_layer = MLP(1024, 1024).to('cuda')
        projection_layer.load_state_dict(torch.load(checkpoint))
    else:
        projection_layer = None
    distances = []

    # iterate over transforms
    for transform_config in transforms:
        if 'resize' in transform_config:
            interp_style = transform_config.split('-')[1]
            size = int(transform_config.split('-')[-1])
            print(size, interp_style)
            if interp_style == 'nearest':
                resize = tf.Resize((size, size), interpolation=Image.NEAREST)
            elif interp_style == 'bilinear':
                resize = tf.Resize((size, size), interpolation=Image.BILINEAR)
            elif interp_style == 'bicubic':
                resize = tf.Resize((size, size), interpolation=Image.BICUBIC)
            elif interp_style == 'lanczos':
                resize = tf.Resize((size, size), interpolation=Image.LANCZOS)
            elif interp_style == 'recon_down':
                #Setting where an image is downsampled first followed by upsizing 
                resize = tf.Compose(
                                    [tf.Resize((int(scale*size), int(scale*size)), interpolation=Image.BICUBIC),
                                    tf.Resize((size, size), interpolation=Image.BICUBIC)]
                        )

            transform = tf.Compose(
                [
                    resize,
                    tf.ToImage(),
                    tf.ToDtype(torch.float32, scale=True),
                ]
            )
        elif 'random' in transform_config:
            transform = RandomAugment()
        elif transform_config != "clean":
            transform = tf.Compose(
                [
                    transform_from_config(transform_config)
                    if isinstance(transform_config, str)
                    else transform_config,
                    tf.ToImage(),
                    tf.ToDtype(torch.float32, scale=True),
                ]
            )


        # iterate over directories
        for idx,dir in enumerate(dirs):
            if transform_config != "clean":
                ds = ImageFolder(dir, amount=amount, transform=transform)
            else:
                ds = ImageFolder(dir, amount=amount)
            print('_______________________________')
            print(dir)
            print('_______________________________')
            # iterate over autoencoder repo_ids
            for repo_id in repo_ids:
                rec_paths = compute_reconstructions(
                    ds,
                    repo_id=repo_id,
                    output_root=reconstruction_root,
                    seed=seed,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    vae_path=vae_path,
                    do_over = do_over,
                    optimize = optimize,
                    iterations = iterations
                    )
                if post_transform:
                    ds_rec = ImageFolder(rec_paths, transform = transform)
                else:
                    ds_rec = ImageFolder(rec_paths)

                # iterate over distance metrics
                for dist_metric in distance_metrics:
                    
                    dist_dict, files = distance_from_config(
                                                            dist_metric,
                                                            batch_size=batch_size,
                                                            num_workers=num_workers,
                                                            spatial = spatial,
                                                            projection_layer=projection_layer,
                                                            **distance_kwargs,
                                                        ).compute(
                                                                    ds_a=ds,
                                                                    ds_b=ds_rec,
                                                                )
                    for dist_name, dist_tensor in dist_dict.items():
                        if not distance_kwargs.get("spatial", False):
                            dist_tensor = dist_tensor.squeeze(1, 2, 3)
                        df = pd.DataFrame(
                            {
                                "dir": str(dir),
                                "image_size": int(ds[0][0].shape[-1]),
                                "repo_id": repo_id,
                                "transform": transform_config,
                                "distance_metric": dist_name,
                                "file": files,
                                "distance": list(dist_tensor.numpy()),
                            }
                        )
                        distances.append(df)
                    pbar.update()

    distances = pd.concat(distances)

    # determine maximum distance over all repo_ids for each file
    if compute_max:
        maxima = []
        for group_keys, group_df in distances.groupby(
            group_cols := ["dir", "image_size", "transform", "distance_metric"],
            sort=False,
        ):
            max_values = group_df.groupby("file").apply(
                lambda df: np.stack(df.distance).max(axis=0)
            )
            max_df = {col: key for col, key in zip(group_cols, group_keys)}
            max_df.update(
                {
                    "repo_id": "max",
                    "file": max_values.index.values,
                    "distance": max_values.values,
                }
            )
            maxima.append(pd.DataFrame(max_df))

        distances = pd.concat([distances, *maxima]).sort_values("dir", kind="stable")
    distances = distances.reset_index(drop=True)
    return distances


def compute_complexities(
    dirs: list[Path],
    transforms: list[str],
    complexity_metrics: list[str],
    amount: Optional[int],
    patch_size: Optional[int],
    patch_stride: Optional[int],
    batch_size: int,
    num_workers: int,
    do_over = True,
) -> pd.DataFrame:
    """Compute distances between original and reconstructed images."""
    # set up progress bar
    pbar = tqdm(
        desc="PROGRESS (compute_complexities)",
        total=len(transforms) * len(dirs) * len(complexity_metrics),
    )

    complexities = []

    # iterate over transforms
    for transform_config in transforms:
        if transform_config != "clean":
            transform = tf.Compose(
                [
                    transform_from_config(transform_config),
                    tf.ToImage(),
                    tf.ToDtype(torch.float32, scale=True),
                ]
            )

        # iterate over directories
        for dir in dirs:
            if transform_config != "clean":
                ds = ImageFolder(dir, amount=amount, transform=transform)
            else:
                ds = ImageFolder(dir, amount=amount)

            # iterate over complexity metrics
            for comp_metric in complexity_metrics:
                comp_dict, files = complexity_from_config(
                    comp_metric,
                    patch_size=patch_size,
                    patch_stride=patch_stride,
                    batch_size=batch_size,
                    num_workers=num_workers,
                ).compute(
                    ds=ds,
                )
                for comp_name, comp_tensor in comp_dict.items():
                    df = pd.DataFrame(
                        {
                            "dir": str(dir),
                            "transform": transform_config,
                            "complexity_metric": comp_name,
                            "file": files,
                            "complexity": list(comp_tensor.numpy()),
                        }
                    )
                    complexities.append(df)
                pbar.update()

    complexities = pd.concat(complexities).reset_index(drop=True)
    return complexities
