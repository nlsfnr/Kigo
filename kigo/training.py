'''The training loop and associated functions.'''
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
import jmp

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
    scale: jmp.LossScale


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
    '''The forward function for computing the loss. The `cfg` argument will be
    baked into the function before JIT compilation using functools.partial.'''
    return Model.from_cfg(cfg)(xt, snr, True)


def loss_fn(params: optax.Params,
            xt: Array,
            noise: Array,
            snr: Array,
            rng: PRNGKey,
            scale: jmp.LossScale,
            *,
            cfg: Config,
            ) -> Tuple[Array, Array]:
    '''The loss function. The `cfg` argument will be baked into the function
    before JIT compilation using functools.partial.'''
    apply = hk.transform(partial(forward_fn, cfg=cfg)).apply
    noise_pred = apply(params, rng, xt, snr)
    loss = jnp.mean((noise_pred - noise) ** 2)
    return scale.scale(loss), loss


def train_step_fn(x0: Array,
                  params: optax.Params,
                  ema: optax.Params,
                  opt_state: optax.OptState,
                  rng: PRNGKey,
                  scale: jmp.LossScale,
                  *,
                  cfg: Config,
                  ) -> Tuple[optax.Params, optax.Params, optax.OptState,
                             Array, jmp.LossScale]:
    '''Performs one training iteration This function is `pmap`ped across all
    devices. To that end, only the gradients are shared between the latter. The
    `cfg` argument will be baked into the function before JIT compilation using
    functools.partial.'''
    rng, rng0, rng1, rng2 = jax.random.split(rng, 4)
    noise = jax.random.normal(rng0, shape=x0.shape)
    snr = cosine_snr(jax.random.uniform(rng1, shape=(len(x0),)))
    xt = sample_q(x0, noise, snr)
    grad_fn = jax.grad(partial(loss_fn, cfg=cfg), has_aux=True)
    gradients, loss = grad_fn(params, xt, noise, snr, rng2, scale)
    gradients = jax.lax.pmean(gradients, axis_name='device')
    gradients = scale.unscale(gradients)
    gradients_finite = jmp.all_finite(gradients)
    scale = scale.adjust(gradients_finite)
    opt = get_opt(cfg)
    updates, new_opt_state = opt.update(gradients, opt_state, params=params)
    new_params = optax.apply_updates(params, updates)
    new_ema = optax.incremental_update(params, ema,
                                       step_size=1 - cfg.tr.ema_alpha)
    # Only actually update the params etc. if all gradients were finite
    opt_state, params, ema = jmp.select_tree(
        gradients_finite,
        (new_opt_state, new_params, new_ema),
        (opt_state, params, ema))
    return params, ema, opt_state, loss ** 0.5, scale


def get_policy(cfg: Config) -> jmp.Policy:
    half = jnp.float16 if cfg.tr.use_fp16 else jnp.float32
    full = jnp.float32
    model_policy = jmp.Policy(param_dtype=full,
                              compute_dtype=half,
                              output_dtype=full)
    return model_policy


def train(params: optax.Params,
          ema: optax.Params,
          opt_state: optax.OptState,
          rngs: hk.PRNGSequence,
          dataset: Dataset,
          workdir: Directory,
          cfg: Config,
          ctx: Context,
          ) -> Iterator[Pack]:
    '''The training loop. Repeatedly calls train_step_fn, yielding the results
    to downstream tasks such as logging etc. To allow training on multiple
    devices, the params, optimizer state etc. have to be broadcasted across
    devices first whilst the batches are divided between them.'''
    p_train_step = jax.pmap(partial(train_step_fn, cfg=cfg),
                            axis_name='device', donate_argnums=5)
    device_count = jax.device_count()
    logger.info(f'Devices found: {device_count}.')
    p_params = pytree_broadcast(params, device_count)
    p_ema = pytree_broadcast(ema, device_count)
    p_opt_state = pytree_broadcast(opt_state, device_count)
    p_scale = pytree_broadcast(
        jmp.DynamicLossScale(
            jnp.array(cfg.tr.loss_scale),
            counter=jnp.array(ctx.iteration % cfg.tr.dynamic_scale_period),
            period=cfg.tr.dynamic_scale_period),
        device_count)
    batch_it = (batch for _ in count()
                for batch in iter(dataset.dataloader()))
    policy = get_policy(cfg)
    hk.mixed_precision.set_policy(Model, policy)
    while True:
        batch = policy.cast_to_compute(next(batch_it))
        p_batch = rearrange(batch, '(d b) ... -> d b ...',
                            d=device_count)
        p_rng = jax.random.split(next(rngs), num=device_count)
        state = p_train_step(p_batch, p_params, p_ema, p_opt_state, p_rng,
                             p_scale)
        p_params, p_ema, p_opt_state, p_mae, p_scale = state
        mae = p_mae.mean()
        ctx.iteration += 1
        yield Pack(workdir=workdir,
                   cfg=cfg,
                   ctx=ctx,
                   params=pytree_collapse(p_params),
                   ema=pytree_collapse(p_ema),
                   opt_state=pytree_collapse(p_opt_state),
                   rngs=rngs,
                   mae=mae,
                   scale=pytree_collapse(p_scale))


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
    pack = None
    is_first = True
    try:
        for pack in packs:
            if is_first:
                logger.info('Started training at iteration '
                            f'{pack.ctx.iteration}')
                is_first = False
            if pack.ctx.periodically(pack.cfg.tr.yield_freq):
                logger.info(f'{pack.ctx.iteration:>6}'
                            f' | MAE: {round(float(pack.mae), 6):>}')
            yield pack
    finally:
        if pack is not None:
            logger.info(f'Training ended at iteration {pack.ctx.iteration}')


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
    d = {'MAE': mean_mae, 'Loss scale': float(pack.scale.loss_scale)}
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
