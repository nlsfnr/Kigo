from __future__ import annotations
from typing import List, Callable, Iterator, Union, Any
import os
import random
from functools import partial
from pathlib import Path
from logging import getLogger
from torch.utils.data import Dataset as PTDataset, DataLoader
import numpy.typing as npt
import numpy as np
from PIL import Image

from .configs import Config, ImageConfig


logger = getLogger('Noisy.dataset')
ArrF32 = npt.NDArray[np.float32]


class Dataset(PTDataset[ArrF32]):

    def __init__(self, cfg: Config,
                 transform: Callable[[Image.Image], ArrF32]
                 ) -> None:
        self.cfg = cfg
        self.paths = self._load_paths()
        self.transform = transform

    @classmethod
    def from_cfg(cls, cfg: Config) -> Dataset:
        return cls(cfg, partial(transform, img_cfg=cfg.img))

    def _load_paths(self) -> List[Path]:
        '''Returns all paths of the files inside self.data_cfg.path that have
        suffixes contained in self.data_cfg.suffixes.'''
        path = Path(self.cfg.ds.path)
        cache_file = path / '.cache.txt'
        if cache_file.exists():
            logger.info(f'Found cache: {cache_file}')
            with open(cache_file) as fh:
                return [Path(s.strip()) for s in fh.readlines()]
        logger.info(f'Loading filenames from {path}. This might take a while.')
        extensions = tuple([e.lower().strip()
                            for e in self.cfg.ds.extensions])
        files = []
        for root_str, _, files_ in os.walk(path):
            root = Path(root_str)
            for file_str in files_:
                if not file_str.lower().endswith(extensions):
                    continue
                file = root / file_str
                files.append(file.resolve())
                if len(files) % 100_000 == 0:
                    logger.info(f'Found {len(files)} files...')
        logger.info(f'Found a total of {len(files)} files in {path}')
        logger.info(f'Storing files in cache: {cache_file}')
        with open(cache_file, 'w') as fh:
            fh.writelines((str(p) + '\n' for p in files))
        return files

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> ArrF32:
        return self.load_image(self.paths[index])

    def load_image(self, path: Path) -> ArrF32:
        img = Image.open(path)
        img_np = self.transform(img)
        return img_np

    def __iter__(self) -> Iterator[ArrF32]:
        return (self[i] for i in range(len(self)))

    def dataloader(self) -> NumpyDataLoader:
        return NumpyDataLoader(self,
                               batch_size=self.cfg.tr.batch_size,
                               shuffle=True,
                               drop_last=True,
                               num_workers=self.cfg.ds.loader_worker_count)

    def sample(self, n: int) -> ArrF32:
        '''Returns random sample of n images.'''
        indices = [random.randint(0, len(self) - 1) for _ in range(n)]
        return np.stack([self[idx] for idx in indices])


def numpy_collate(batch: Union[List[ArrF32], ArrF32]) -> ArrF32:
    if isinstance(batch, list):
        return np.stack(batch)
    return batch


class NumpyDataLoader(DataLoader[ArrF32]):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.pop('collate_fn', None)
        super().__init__(*args, collate_fn=numpy_collate,  # type: ignore
                         **kwargs)


def transform(img: Image.Image, img_cfg: ImageConfig
              ) -> ArrF32:
    side = min(img.width, img.height) // 2
    x = (img.width // 2)
    y = (img.height // 2)
    # Center crop
    img = img.crop((x - side, y - side, x + side, y + side))
    # Resize
    img = img.resize((img_cfg.size, img_cfg.size))
    # Normalize to -1, 1
    img_np = np.array(img).astype(np.float32) / 127.5 - 1.
    return img_np
