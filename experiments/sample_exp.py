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
from aeroblade.data import ImageFolder, read_files
from aeroblade.distances import distance_from_config
from aeroblade.image import compute_reconstructions
from aeroblade.transforms import transform_from_config
amt=30
#ds = ImageFolder(Path('/nobackup3/anirudh/aeroblade/data/raw/real/00000'), amount=amt)
#ds_f = ImageFolder(Path('/nobackup3/anirudh/aeroblade/data/raw/real/00000'), amount=5)
#ds_pre_rec = ImageFolder(Path('/nobackup3/anirudh/aeroblade/data/pre_opt/7efa52dc37c1aeeb52ab8b866eb335d9/1'), amount=amt)
#ds_post_rec = ImageFolder(Path('/nobackup3/anirudh/aeroblade/data/reconstructions/7efa52dc37c1aeeb52ab8b866eb335d9/1'), amount=amt)

dsf = ImageFolder(Path('/nobackup3/anirudh/datasets/laion_generated_images/1_fake_train'), amount=amt)
#ds_f = ImageFolder(Path('/nobackup3/anirudh/aeroblade/data/raw/real/00000'), amount=5)
dsf_pre_rec = ImageFolder(Path('/nobackup3/anirudh/aeroblade/data/pre_opt/fdd1131c10cc5d92be932a59fa53981f/1'), amount=amt)
#dsf_post_rec = ImageFolder(Path('/nobackup3/anirudh/aeroblade/data/reconstructions/fdd1131c10cc5d92be932a59fa53981f/1'))
dsf_post_recnoamt = ImageFolder(read_files(Path('data/reconstructions/fdd1131c10cc5d92be932a59fa53981f/1')))
dsf_post_rec = ImageFolder(read_files(Path('data/reconstructions/fdd1131c10cc5d92be932a59fa53981f/1')), amount=amt)
dist_dict_1, files = distance_from_config(
                        config="lpips_vgg_-1",
                        batch_size=5,
                        num_workers=4,
                    ).compute(
                        ds_a=dsf,
                        ds_b=dsf_pre_rec,
                        retSpat=False
                    )
dist_dict_2, files = distance_from_config(
                        config="lpips_vgg_-1",
                        batch_size=5,
                        num_workers=4,
                    ).compute(
                        ds_a=dsf_post_rec,
                        ds_b=dsf_post_recnoamt,
                        retSpat=False
                    )


for key in dist_dict_1.keys():
    print(key)
    for idx in range(len(dist_dict_1[key])):
        print(dist_dict_1[key][idx], dist_dict_2[key][idx])
