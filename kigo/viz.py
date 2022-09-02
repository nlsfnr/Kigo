from typing import Union, Optional
from pathlib import Path
import math
from chex import Array
import matplotlib.pyplot as plt
import matplotlib.image as plt_img
import numpy as np
import numpy.typing as npt


IMGLike = Union[npt.NDArray[np.float32], npt.NDArray[np.int32], Array]


def canonicalize(imgs: IMGLike) -> npt.NDArray[np.float32]:
    # Canonicalize imgs, s.t. 0 <= imgs <= 1 and imgs.shape == b, h, w, c
    imgs_ = np.array(imgs)
    if imgs_.dtype == np.uint8:
        imgs_ = imgs_.astype(np.float32) / 256.
    else:
        imgs_ = imgs_ * 0.5 + 0.5
    imgs_ = imgs_.clip(0., 1.)
    if len(imgs_.shape) == 3:
        # Add batch dimension
        imgs_ = np.expand_dims(imgs_, 0)
    return imgs_


def show(imgs: IMGLike,
         n_cols: Optional[int] = None,
         out: Optional[Path] = None,
         show: bool = True,
         axis: Optional[plt.Axes] = None,
         ) -> None:
    imgs_ = canonicalize(imgs)
    grid = make_grid(imgs_, n_cols)
    # Display
    if axis is None:
        axis = plt.gca()
    assert axis is not None
    if out is None:
        axis.imshow(grid)
        axis.axis('off')
        if show:
            plt.tight_layout()
            plt.show()
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        plt_img.imsave(out, grid)


def make_grid(imgs: IMGLike, n_cols: Optional[int] = None
              ) -> npt.NDArray[np.float32]:
    msg = f'{imgs.min()=}, {imgs.max()=}, {imgs.mean()=}, {imgs.std()=}'
    assert (0. <= imgs).all() and (imgs <= 1.).all(), msg
    b, h, w, c = imgs.shape
    n_cols = n_cols or min(b, 8)
    n_rows = math.ceil(imgs.shape[0] / n_cols)
    cols = np.array_split(imgs, n_cols)
    cols = [col.reshape(h * col.shape[0], w, c) for col in cols]
    cols = [np.pad(col, ((0, h * n_rows - col.shape[0]), (0, 0), (0, 0)))
            for col in cols]
    grid = np.concatenate(cols, axis=1)
    return grid
