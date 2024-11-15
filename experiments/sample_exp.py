import argparse

from pathlib import Path

import pandas as pd
from aeroblade.evaluation import tpr_at_max_fpr
from aeroblade.high_level_funcs import compute_distances
from aeroblade.misc import safe_mkdir, write_config
from sklearn.metrics import average_precision_score
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as tf
from tqdm import tqdm

from aeroblade.complexities import complexity_from_config
from aeroblade.data import ImageFolder, read_files, get_all_files
from aeroblade.distances import distance_from_config
from aeroblade.image import compute_reconstructions
from aeroblade.transforms import transform_from_config
amt=20
#ds = ImageFolder(Path('/nobackup3/anirudh/aeroblade/data/raw/real/00000'), amount=amt)
#ds_f = ImageFolder(Path('/nobackup3/anirudh/aeroblade/data/raw/real/00000'), amount=5)
#ds_pre_rec = ImageFolder(Path('/nobackup3/anirudh/aeroblade/data/pre_opt/7efa52dc37c1aeeb52ab8b866eb335d9/1'), amount=amt)
#ds_post_rec = ImageFolder(Path('/nobackup3/anirudh/aeroblade/data/reconstructions/7efa52dc37c1aeeb52ab8b866eb335d9/1'), amount=amt)
if amt == 800:
    ds_ip2p = ImageFolder(Path('/nobackup3/anirudh/datasets/instructpix2pix_dataset'), amount=amt)
    ds_ip2p_rec = ImageFolder(read_files(Path('/nobackup3/anirudh/aeroblade/data/reconstructions/e2ea2de35667116c8c33935715d8592e/1')), amount=amt)
dsf = ImageFolder(Path('/nobackup3/anirudh/datasets/laion_generated_images/1_fake_val'), amount=amt)
#ds_f = ImageFolder(Path('/nobackup3/anirudh/aeroblade/data/raw/real/00000'), amount=5)
dsf_pre_rec = ImageFolder(Path('/nobackup3/anirudh/aeroblade/data/pre_valopt/71a6b96c50492488a9479bf173920411/1'), amount=amt)
#dsf_post_rec = ImageFolder(Path('/nobackup3/anirudh/aeroblade/data/post_valopt/71a6b96c50492488a9479bf173920411/1'))
#dsf_post_recnoamt = ImageFolder(read_files(Path('data/reconstructions/71a6b96c50492488a9479bf173920411/1')))
dsf_post_rec = ImageFolder(read_files(Path('data/post_valopt/71a6b96c50492488a9479bf173920411/1')), amount=amt)
if True:
    dist_dict_1, files = distance_from_config(
                        config="lpips_vgg_-1",
                        batch_size=5,
                        num_workers=4,
                    ).compute(
                        ds_a=dsf,
                        ds_b=dsf_pre_rec,
                    )
    dist_dict_2, files = distance_from_config(
                        config="lpips_vgg_-1",
                        batch_size=5,
                        num_workers=4,
                    ).compute(
                        ds_a=dsf,
                        ds_b=dsf_post_rec,
                    )
    for key in dist_dict_1.keys():
        print(key)
        if 'vgg_2' not in key:
            continue
        for idx in range(len(dist_dict_1[key])):
            print(dist_dict_1[key][idx],dist_dict_2[key][idx])
else:
    dist_ip2p, files = distance_from_config(
        
                        config="lpips_vgg_-1",
                        batch_size=20,
                        num_workers=4,
                    ).compute(
                        ds_a=ds_ip2p,
                        ds_b=ds_ip2p_rec,
                    )

    for key in dist_ip2p.keys():
        print(key)
        if 'vgg_2' not in key:
            continue
        for idx in range(len(dist_ip2p[key])):
            print(dist_ip2p[key][idx])
