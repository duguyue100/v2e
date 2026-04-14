"""customized Pytorch dataloader

@author: Zhe He
@contact: zhehe@student.ethz.ch
@latest update: 2019-May-27th
"""

import glob
from pathlib import Path
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch.utils.data as data
from PIL import Image


class Frames(data.Dataset[Any]):  # type: ignore[misc]
    """
    Load frames from an N-d array, and transform them into tensor.
    @Author:
        - Zhe He
        - zhehe@student.ethz.ch

    @Members:
        array: N-d numpy array.
        transform: Compose object.

    @Methods:
        __getitem__: List(Tensor, Tensor)
            return a pair of (frame0, frame1).
        __len__: int
            return the length of the dataset.
        __repr__: str
            return printable representation of the class.
    """

    def __init__(
        self, array: Any, transform: Optional[Callable[..., Any]] = None
    ) -> None:
        """
        Args:
                array: N-d numpy array.
                transform: Compose object.
        """
        self.array = array
        self.transform = transform
        self.origDim = array.shape[2], array.shape[1]
        self.dim = (int(self.origDim[0] / 32) * 32, int(self.origDim[1] / 32) * 32)

    def __getitem__(self, index: int) -> List[Any]:
        """Return an item from the dataset.

        @Parameter:
            index: int.
        @Return: List(Tensor, Tensor).
        """
        sample = []
        # Loop over for all frames corresponding to the `index`.
        for image_array in [self.array[index], self.array[index + 1]]:
            # Open image using pil.
            image = Image.fromarray(image_array)
            image = image.resize(self.dim, Image.Resampling.LANCZOS)
            # Apply transformation if specified.
            if self.transform is not None:
                image = self.transform(image)
            sample.append(image)
        return sample

    def __len__(self) -> int:
        """Return the size of the dataset.
        @Return: int.
        """
        return int(self.array.shape[0]) - 1

    def __repr__(self) -> str:
        """Return printable representations of the class.
        @Return: str.
        """
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += f"    Number of datapoints: {self.__len__()}\n"
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str


class FramesDirectory(data.Dataset[Any]):  # type: ignore[misc]
    """
    Load frames from a directory that has individual frame records,
    and transform them into tensor.
    """

    def __init__(
        self,
        folder_path: Path,
        ori_dim: Tuple[int, int],
        parsing: str = "/*.npy",
        transform: Optional[Callable[..., Any]] = None,
    ) -> None:
        """
        Args:
                array: N-d numpy array.
                transform: Compose object.
        """
        self.files = sorted(
            glob.glob(f"{folder_path}" + parsing),
            key=lambda p: int(Path(p).stem) if Path(p).stem.isdigit() else p,
        )

        self.transform = transform
        self.origDim = ori_dim
        #  self.origDim = array.shape[2], array.shape[1]
        self.dim = (int(self.origDim[0] / 32) * 32, int(self.origDim[1] / 32) * 32)

    def __getitem__(self, index: int) -> List[Any]:
        """Return an item from the dataset.

        @Parameter:
            index: int.
        @Return: List(Tensor, Tensor).
        """
        sample = []

        image_1 = np.load(self.files[index])
        image_2 = np.load(self.files[index + 1])
        # Loop over for all frames corresponding to the `index`.
        for image_array in [image_1, image_2]:
            # Open image using pil.
            image = Image.fromarray(image_array)
            image = image.resize(self.dim, Image.Resampling.LANCZOS)
            # Apply transformation if specified.
            if self.transform is not None:
                image = self.transform(image)
            sample.append(image)
        return sample

    def __len__(self) -> int:
        """Return the size of the dataset.
        @Return: int.
        """
        return len(self.files) - 1

    def __repr__(self) -> str:
        """Return printable representations of the class.
        @Return: str.
        """
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += f"    Number of datapoints: {self.__len__()}\n"
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str
