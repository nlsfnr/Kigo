'''Functions to save and load Kigo components to and from directories, called
checkpoints.'''
from __future__ import annotations
from typing import Optional, Union
from collections.abc import Mapping
from pathlib import Path
from dataclasses import asdict
import yaml
import pickle
import jax
import numpy as np
import haiku as hk
import optax

from .configs import Config
from .utils import get_logger, make_symlink, PathLike, Directory, File
from .nn import get_params_and_forward_fn
from . import training


logger = get_logger('kigo.persistence')


CONFIG_FILE = 'config.yaml'
PARAMS_FILE = 'model-params.pkl'
EMA_FILE = 'ema-params.pkl'
OPT_STATE_FILE = 'optimizer-state.pkl'
CTX_FILE = 'context.yaml'

LATEST_CHECKPOINT_SYMLINK = 'latest-checkpoint'
WORKDIR_MARKER = 'workdir-marker'
CHECKPOINT_MARKER = 'checkpoint-marker'
INITIAL_CHECKPOINT_NAME = 'initial'


Params = Union[hk.Params, optax.Params]


def get_config(x: Union[PathLike, Config]) -> Config:
    if isinstance(x, Config):
        return x
    else:
        path = File(x)
    return Config.from_yaml(path)


def get_workdir(x: PathLike) -> Directory:
    path = Directory(x)
    if is_checkpoint(path):
        return get_workdir(path.parent)
    if not is_workdir(path):
        raise IOError(f'Not a workdir: {path} (missing {WORKDIR_MARKER} file)')
    return path


def is_workdir(path: PathLike) -> bool:
    path = Directory(path)
    return (path / WORKDIR_MARKER).exists()


def mark_as_workdir(path: PathLike) -> Directory:
    path = Directory.ensure(path)
    if not is_workdir(path):
        (path / WORKDIR_MARKER).touch()
    return path


def get_checkpoint(x: PathLike) -> Directory:
    path = Directory(x)
    if is_workdir(path):
        latest_cp = path / LATEST_CHECKPOINT_SYMLINK
        return get_checkpoint(latest_cp.resolve())
    if not is_checkpoint(path):
        raise IOError(f'Neither a checkpoint nor a workdir: {path} (missing '
                      f'{WORKDIR_MARKER} or {CHECKPOINT_MARKER} file)')
    return path


def is_checkpoint(path: PathLike) -> bool:
    path = Directory(path)
    return (path / CHECKPOINT_MARKER).exists()


def mark_as_checkpoint(path: Path, update_latest: bool = True) -> Directory:
    path = Directory.ensure(path)
    if not is_checkpoint(path):
        (path / CHECKPOINT_MARKER).touch()
    if update_latest:
        wd = path.parent
        if not is_workdir(wd):
            raise IOError(f'Refusing to add symlink to non-workdir: {wd}')
        make_symlink(path, wd / LATEST_CHECKPOINT_SYMLINK)
    return path


def init_workdir(workdir: Path, config_file: File, rng_key: hk.PRNGSequence
                 ) -> None:
    workdir = Directory.ensure(workdir)
    mark_as_workdir(workdir)
    cfg = Config.from_yaml(config_file)
    params, _ = get_params_and_forward_fn(cfg, rng_key)
    _, opt_state = training.get_opt_and_opt_state(cfg, params)
    ctx = training.Context()
    save(workdir / INITIAL_CHECKPOINT_NAME,
         params=params,
         ema=params,
         cfg=cfg,
         opt_state=opt_state,
         ctx=ctx)
    logger.info(f'Initialized a new checkpoint in {workdir}')
    n_params = sum(np.prod(np.array(leaf.shape))
                   for leaf in jax.tree_util.tree_leaves(params))
    logger.info(f'The model has {round(n_params / 1e6, 2)}M parameters')


def save(cp: Path,
         *,
         cfg: Optional[Config] = None,
         params: Optional[Params] = None,
         ema: Optional[Params] = None,
         opt_state: Optional[optax.OptState] = None,
         ctx: Optional[training.Context] = None,
         update_latest: bool = True,
         ) -> None:
    cp = Directory.ensure(cp)
    if cfg is not None:
        cfg.to_yaml(cp / CONFIG_FILE)
    if params is not None:
        with open(cp / PARAMS_FILE, 'wb') as fh:
            pickle.dump(params, fh)
    if ema is not None:
        with open(cp / EMA_FILE, 'wb') as fh:
            pickle.dump(ema, fh)
    if opt_state is not None:
        with open(cp / OPT_STATE_FILE, 'wb') as fh:
            pickle.dump(opt_state, fh)
    if ctx is not None:
        with open(cp / CTX_FILE, 'w') as fh:
            yaml.safe_dump(asdict(ctx), fh)
    mark_as_checkpoint(cp, update_latest=update_latest)
    logger.info(f'Saved to {cp}')


def load_cfg(cp: Path) -> Config:
    return Config.from_yaml(File(cp / CONFIG_FILE))


def load_params(cp: Path) -> Params:
    with open(cp / PARAMS_FILE, 'rb') as fh:
        params = pickle.load(fh)
    assert isinstance(params, Mapping)
    return params


def load_ema(cp: Path) -> Params:
    with open(cp / EMA_FILE, 'rb') as fh:
        ema = pickle.load(fh)
    assert isinstance(ema, Mapping)
    return ema


def load_opt_state(cp: Path) -> optax.OptState:
    with open(cp / OPT_STATE_FILE, 'rb') as fh:
        opt_state = pickle.load(fh)
    return opt_state


def load_ctx(cp: Path) -> training.Context:
    with open(cp / CTX_FILE) as fh:
        data = yaml.safe_load(fh)
    assert isinstance(data, dict)
    return training.Context(**data)
