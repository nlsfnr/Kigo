from __future__ import annotations
from typing import Iterator, Optional, Tuple
from functools import partial
from itertools import count
from dataclasses import dataclass
from pprint import pformat
import optax
import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp
from chex import Array, PRNGKey
from einops import rearrange
import wandb
from wandb.sdk.wandb_run import Run as WandbRun

from .utils import (Directory, get_logger, pytree_broadcast, pytree_collapse,
                    random_name)
from .configs import Config
from .nn import Model, get_params_and_forward_fn
from .data import Dataset
from .diffusion import sample_q, sample_p, cosine_snr
from . import persistence


logger = get_logger('kigo.training')


@dataclass
class Context:
    iteration: int = 0
    wandb_run_id: Optional[str] = None

    def periodically(self, freq: int, skip_first: bool = True) -> bool:
        if skip_first and self.iteration == 0:
            return False
        return self.iteration % freq == 0

    @classmethod
    def from_cfg(cls, _: Config) -> Context:
        return cls()


@dataclass
class Pack:
    workdir: Directory
    cfg: Config
    ctx: Context
    params: optax.Params
    ema: optax.Params
    opt_state: optax.OptState
    rngs: hk.PRNGSequence
    mae: float


def get_opt(cfg: Config) -> optax.GradientTransformation:

    def lr_schedule(step: Array) -> Array:
        return jnp.minimum(1., step / cfg.tr.learning_rate_warmup_steps)

    opt = optax.chain(
        optax.clip(cfg.tr.gradient_clipping),
        optax.adamw(learning_rate=cfg.tr.learning_rate,
                    weight_decay=cfg.tr.weight_decay),
        optax.scale_by_schedule(lr_schedule),
    )
    return opt


def forward_fn(xt: Array,
               snr: Array,
               *,
               cfg: Config,
               ) -> Array:
    return Model.from_cfg(cfg)(xt, snr, True)


def loss_fn(params: optax.Params,
            xt: Array,
            noise: Array,
            snr: Array,
            rng: PRNGKey,
            *,
            cfg: Config,
            ) -> Tuple[Array, Array]:
    apply = hk.transform(partial(forward_fn, cfg=cfg)).apply
    noise_pred = apply(params, rng, xt, snr)
    loss = jnp.mean((noise_pred - noise) ** 2)
    return loss, loss


def train_step_fn(x0: Array,
                  params: optax.Params,
                  ema: optax.Params,
                  opt_state: optax.OptState,
                  rng: PRNGKey,
                  *,
                  cfg: Config,
                  ) -> Tuple[optax.Params, optax.Params, optax.OptState,
                             Array]:
    rng, rng_split = jax.random.split(rng)
    noise = jax.random.normal(rng_split, shape=x0.shape)
    rng, rng_split = jax.random.split(rng)
    snr = cosine_snr(jax.random.uniform(rng_split, shape=(len(x0),)))
    xt = sample_q(x0, noise, snr)
    rng, rng_split = jax.random.split(rng)
    grad_fn = jax.grad(partial(loss_fn, cfg=cfg), has_aux=True)
    gradients, loss = grad_fn(params, xt, noise, snr, rng_split)
    gradients = jax.lax.pmean(gradients, axis_name='device')
    opt = get_opt(cfg)
    updates, opt_state = opt.update(gradients, opt_state, params=params)
    params = optax.apply_updates(params, updates)
    ema = optax.incremental_update(params, ema, step_size=1 - cfg.tr.ema_alpha)
    return params, ema, opt_state, loss ** 0.5


def train(params: optax.Params,
          ema: optax.Params,
          opt_state: optax.OptState,
          rngs: hk.PRNGSequence,
          dataset: Dataset,
          workdir: Directory,
          cfg: Config,
          ctx: Context,
          ) -> Iterator[Pack]:
    p_train_step = jax.pmap(partial(train_step_fn, cfg=cfg),
                            axis_name='device', donate_argnums=5)
    device_count = jax.device_count()
    logger.info(f'Devices found: {device_count}.')
    p_params = pytree_broadcast(params, device_count)
    p_ema = pytree_broadcast(ema, device_count)
    p_opt_state = pytree_broadcast(opt_state, device_count)
    batch_it = (batch for _ in count()
                for batch in iter(dataset.dataloader()))
    while True:
        p_batch = rearrange(next(batch_it), '(d b) ... -> d b ...',
                            d=device_count)
        p_rng = jax.random.split(next(rngs), num=device_count)
        state = p_train_step(p_batch, p_params, p_ema, p_opt_state, p_rng)
        p_params, p_ema, p_opt_state, p_mae = state
        mae = p_mae.mean()
        ctx.iteration += 1
        yield Pack(workdir=workdir,
                   cfg=cfg,
                   ctx=ctx,
                   params=pytree_collapse(p_params),
                   ema=pytree_collapse(p_ema),
                   opt_state=pytree_collapse(p_opt_state),
                   rngs=rngs,
                   mae=mae)


def autosave(packs: Iterator[Pack]) -> Iterator[Pack]:

    def _save_pack(cp: Directory, pack: Pack) -> None:
        persistence.save(cp,
                         cfg=pack.cfg,
                         params=pack.params,
                         ema=pack.ema,
                         opt_state=pack.opt_state,
                         ctx=pack.ctx)

    pack = None
    try:
        for pack in packs:
            if pack.ctx.periodically(pack.cfg.tr.save_checkpoint_freq):
                cp = pack.workdir / str(pack.ctx.iteration).zfill(6)
                _save_pack(cp, pack)
            if pack.ctx.periodically(pack.cfg.tr.save_freq):
                _save_pack(pack.workdir / 'latest', pack)
            yield pack
    finally:
        if pack is not None:
            _save_pack(pack.workdir / 'latest', pack)


def log(packs: Iterator[Pack]) -> Iterator[Pack]:
    for pack in packs:
        if pack.ctx.periodically(pack.cfg.tr.yield_freq):
            logger.info(f'{pack.ctx.iteration:>6}'
                        f' | MAE: {round(float(pack.mae), 6):>}')
        yield pack


def wandb_log(packs: Iterator[Pack]) -> Iterator[Pack]:
    run: Optional[WandbRun] = None
    try:
        maes = []
        for pack in packs:
            maes.append(pack.mae)
            if run is None:
                run = _get_wandb_run(pack)
            if pack.ctx.periodically(pack.cfg.tr.yield_freq):
                maes_: Array = jnp.array(maes)
                maes.clear()
                _log_to_wandb(run, pack, jnp.mean(maes_))
            if pack.ctx.periodically(pack.cfg.tr.wandb_.img_freq):
                _img_to_wandb(run, pack)
            yield pack
    finally:
        if run is not None:
            run.finish()


def _get_wandb_run(pack: Pack) -> WandbRun:
    wandb_cfg = pack.cfg.tr.wandb_
    if pack.ctx.wandb_run_id is None:
        run = wandb.init(project=wandb_cfg.project,
                         group=wandb_cfg.group,
                         name=(wandb_cfg.name
                               or random_name(prefix='wandb')),
                         tags=wandb_cfg.tags,
                         resume=False,
                         notes=(f'\nConfig:\n{pformat(pack.cfg.dict())}'))
    else:
        run = wandb.init(id=pack.ctx.wandb_run_id,
                         project=wandb_cfg.project,
                         group=wandb_cfg.group,
                         tags=wandb_cfg.tags,
                         resume=True)
    assert isinstance(run, WandbRun)
    pack.ctx.wandb_run_id = run.id
    logger.info(f'Loaded WandB run with id: {run.id}, name: {run.name} and '
                f'URL: {run.get_url()}')
    return run


def _log_to_wandb(run: WandbRun, pack: Pack, mean_mae: float) -> None:
    d = {'MAE': mean_mae}
    run.log(d, step=pack.ctx.iteration)


def _img_to_wandb(run: WandbRun, pack: Pack) -> None:
    logger.info('Logging images to WandB...')
    wandb_cfg = pack.cfg.tr.wandb_
    n = pack.cfg.tr.wandb_.img_n
    c, s = pack.cfg.img.channels, pack.cfg.img.size
    _, forward_fn = get_params_and_forward_fn(pack.cfg, pack.rngs,
                                              params=pack.ema)
    xT = jax.random.normal(next(pack.rngs), shape=(n, s, s, c))
    x0 = sample_p(xT, forward_fn, wandb_cfg.img_steps, next(pack.rngs),
                  wandb_cfg.img_eta, wandb_cfg.img_clip_percentile)
    log_dict = {'Samples': [wandb.Image(np.array(img)) for img in x0]}
    run.log(log_dict, step=pack.ctx.iteration)
    logger.info('Done')
