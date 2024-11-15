import pdb
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import os
import random
import torch
import torchvision.transforms.v2 as tf
from PIL import Image
from torchvision.datasets import VisionDataset
#pdb.set_trace()
IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]


class ImageFolder(VisionDataset):
    """
    Dataset for reading images from a list of paths, directories, or a mixture of both.
    """

    def __init__(
        self,
        paths: Union[list[Path], Path],
        transform: Optional[Callable] = tf.Compose(
            [tf.ToImage(), tf.ToDtype(torch.float32, scale=True)]
        ),
        amount: Optional[int] = None,
    ) -> None:
        self.paths = [paths] if isinstance(paths, Path) else paths
        self.transform = transform
        self.amount = amount

        self.img_paths = []
        for path in self.paths:
            if path.is_dir():
                for file in get_all_files(path):
                    if file.suffix.lower() in IMG_EXTENSIONS:
                        self.img_paths.append(file)
                        if (
                            self.amount is not None
                            and len(self.img_paths) == self.amount
                        ):
                            break
            else:
                self.img_paths.append(path)
        if self.amount is not None and len(self.img_paths) < self.amount:
            raise ValueError("Number of images is less than 'amount'.")
        self.img_paths = sorted(self.img_paths, key=lambda p: p.name)

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[str, float]]:
        img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img, str(self.img_paths[idx])

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        body.append(f"Paths: {self.paths}")
        body.append(f"Transform: {repr(self.transform)}")
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

def get_all_files(directory_path, amount=800):
    file_paths = []

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            full_path = os.path.join(root, file)
            file_paths.append(Path(full_path))
    return sorted(file_paths)
                                                                                    


def read_files(path: Path) -> list[Path]:
    return sorted(path.iterdir())

