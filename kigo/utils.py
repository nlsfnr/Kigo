from __future__ import annotations
from typing import Union, TypeVar, Type, cast
from abc import ABC, abstractmethod
import os
import logging
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    '''This is quite a headache. Jax uses Abseil (absl) for logging. Abseil is
    quite agressive in taking control of python's logging, so we have to go to
    extra lengths to configure logging as we want it. Luckily, there are
    attempts to fix this: https://github.com/google/jax/pull/10968'''
    handler = logging.StreamHandler()
    fmt = '[%(asctime)s|%(name)s|%(levelname)s] %(message)s'
    handler.setFormatter(logging.Formatter(fmt=fmt))
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def random_name(namespace: str = 'global', prefix: str = 'run') -> str:
    counter_file = Path(f'.{namespace.strip()}.name_counter')
    if counter_file.exists():
        with open(counter_file, 'r') as fh:
            number = int(fh.read())
    else:
        number = 0
    with open(counter_file, 'w') as fh:
        fh.write(str(number + 1))
    return prefix + '-' + str(number).zfill(4)


def make_symlink(dest: Path, src: Path) -> None:
    '''Creates a symlink from src to dest.'''
    if src.exists():
        src.unlink()
    relative_dest = relative_to(dest.absolute(),
                                src.absolute().parent)
    src.symlink_to(relative_dest)


def relative_to(path: Path, root: Path) -> Path:
    return Path(os.path.relpath(path, root))


def relative_to_pwd(path: Path) -> Path:
    return relative_to(path, Path(os.getcwd()))


PathLike = Union[str, Path]
_PathT = TypeVar('_PathT', bound='_ValidatedPath')


class _ValidatedPath(ABC):
    '''pathlib.Path can not be directly inherited from due to its dynamic
    __new__ function that adapts the underlying type to the host OS. Thus, we
    need some behind-the-scenes magic to make validation work. Inspired by:
    https://gist.github.com/mawillcockson/9cdce0e64e1437f8823613dda892b4fc'''

    def __new__(cls: Type[_PathT], value: PathLike) -> _PathT:
        return cast(_PathT, cls.validate(value))

    @staticmethod
    @abstractmethod
    def validate(value: PathLike) -> Path:
        pass


class File(_ValidatedPath, Path):

    @staticmethod
    def validate(value: PathLike) -> Path:
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f'Does not exist: {path}')
        if not path.is_file():
            raise ValueError(f'Not a file: {path}')
        return path

    @classmethod
    def ensure(cls, path: PathLike) -> File:
        path = Path(path)
        if not path.exists():
            path.touch()
        return cls(path)


class Directory(_ValidatedPath, Path):

    @staticmethod
    def validate(value: PathLike) -> Path:
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f'Does not exist: {path}')
        if not path.is_dir():
            raise ValueError(f'Not a directory: {path}')
        return path

    @classmethod
    def ensure(cls, path: PathLike) -> Directory:
        path = Path(path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return cls(path)