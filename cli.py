#!/usr/bin/env python3
from typing import Optional
from pathlib import Path
import warnings
# The warnings come from upstream (Haiku) and would only clutter stderr.
warnings.filterwarnings('ignore')
import random
import click
import haiku as hk
import jax

from kigo.utils import File, Directory, get_logger
from kigo import nn
from kigo import diffusion
from kigo import viz
from kigo import persistence
from kigo import training
from kigo import data


logger = get_logger('kigo.cli')


@click.group('Kigo')
def cli() -> None:
    '''A diffusion model written with DM-Haiku!'''
    pass


@cli.command('init')
@click.argument('workdir', type=Path)
@click.argument('config', type=File)
@click.option('--seed', type=int, default=None)
def init(workdir: Path, config: File, seed: Optional[int]) -> None:
    rngs = get_rngs(seed)
    persistence.init_workdir(workdir, config, rngs)


@cli.command('train')
@click.argument('checkpoint', type=Directory)
@click.option('--debug', is_flag=True)
@click.option('--seed', type=int, default=None)
def train(checkpoint: Path, debug: bool, seed: Optional[int]) -> None:
    jax.config.update('jax_disable_jit', debug)  # type: ignore
    checkpoint = persistence.get_checkpoint(checkpoint)
    rngs = get_rngs(seed)
    cfg = persistence.load_cfg(checkpoint)
    params = persistence.load_params(checkpoint)
    ema = persistence.load_ema(checkpoint)
    opt_state = persistence.load_opt_state(checkpoint)
    dataset = data.Dataset.from_cfg(cfg)
    workdir = persistence.get_workdir(checkpoint)
    ctx = persistence.load_ctx(checkpoint)
    packs = training.train(params, ema, opt_state, rngs, dataset, workdir, cfg,
                           ctx)
    packs = training.autosave(packs)
    if cfg.tr.wandb:
        packs = training.wandb_log(packs)
    packs = training.log(packs)
    for pack in packs:
        del pack


@cli.command('syn')
@click.argument('checkpoint', type=Directory)
@click.option('--steps', type=int, default=64)
@click.option('--no-ema', is_flag=True)
@click.option('--eta', type=float, default=0.)
@click.option('--clip-percentile', '-c', type=float, default=0.995)
@click.option('--out', '-o', type=Path, default=None)
@click.option('--debug', is_flag=True)
@click.option('--seed', type=int, default=None)
def syn(checkpoint: Path, steps: int, no_ema: bool, eta: float,
        clip_percentile: float, out: Optional[Path], seed: Optional[int],
        debug: bool) -> None:
    jax.config.update('jax_disable_jit', debug)  # type: ignore
    checkpoint = persistence.get_checkpoint(checkpoint)
    rngs = get_rngs(seed)
    cfg = persistence.load_cfg(checkpoint)
    params = (persistence.load_params(checkpoint)
              if no_ema else
              persistence.load_ema(checkpoint))
    _, forward_fn = nn.get_params_and_forward_fn(cfg, rngs, params)
    xT = jax.random.normal(next(rngs), shape=(1, *cfg.img.shape))
    x0 = diffusion.sample_p(xT, forward_fn, steps, next(rngs), eta,
                            clip_percentile)
    viz.show(x0, out=out)


def get_rngs(seed: Optional[int]) -> hk.PRNGSequence:
    if seed is None:
        seed = random.randint(0, 1 << 32)
    logger.info(f'Using seed: {seed}')
    return hk.PRNGSequence(seed)


if __name__ == '__main__':
    cli()