from typing import Iterator, Any, Optional, Tuple
from dataclasses import dataclass
from functools import partial
from itertools import count
from pprint import pformat
import haiku as hk
import optax
import jmp
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
import wandb
from wandb.sdk.wandb_run import Run as WandbRun

from .diffusion import sample_q, sample_p, cosine_snr
from .utils import Directory, get_logger, random_name
from .configs import Config
from .data import Dataset
from .nn import (Model, SinusoidalEmbedding, Attention,
                 get_params_and_forward_fn)
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


def get_opt_and_opt_state(cfg: Config,
                          params: optax.Params,
                          opt_state: Optional[optax.OptState] = None,
                          ) -> Tuple[optax.MultiSteps, optax.OptState]:
    opt = optax.adamw(learning_rate=cfg.tr.learning_rate,
                      weight_decay=cfg.tr.weight_decay)
    ms_opt = optax.MultiSteps(opt, cfg.tr.gradient_accumulation_steps)
    opt_state = ms_opt.init(params) if opt_state is None else opt_state
    return ms_opt, opt_state


def train(params: optax.Params,
          ema: optax.Params,
          opt_state: optax.OptState,
          rngs: hk.PRNGSequence,
          dataset: Dataset,
          workdir: Directory,
          cfg: Config,
          ctx: Context,
          ) -> Iterator[Pack]:

    scale = jmp.DynamicLossScale(jnp.asarray(2 ** 15))

    def forward_fn(x0: jnp.ndarray,
                   snr: jnp.ndarray,
                   ) -> jnp.ndarray:
        return Model.from_cfg(cfg)(x0, snr, False)

    forward = hk.transform(forward_fn)

    def loss_fn(params_: optax.Params,
                xt: jnp.ndarray,
                snr: jnp.ndarray,
                noise: jnp.ndarray,
                scale_: jmp.LossScale,
                rng: Any,
                ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jmp.LossScale]]:
        noise_pred = forward.apply(params_, rng, xt, snr)
        loss = jnp.mean((noise_pred - noise) ** 2)
        loss_scaled = scale_.scale(loss)
        return loss_scaled, (loss, scale)

    gradient_fn = jax.grad(loss_fn, has_aux=True)
    opt, opt_state = get_opt_and_opt_state(cfg, params, opt_state)

    # Set policies for mixed precision
    half, full = jnp.float16 if cfg.tr.use_fp16 else jnp.float32, jnp.float32
    hk.mixed_precision.set_policy(SinusoidalEmbedding,
                                  jmp.Policy(param_dtype=full,
                                             compute_dtype=full,
                                             output_dtype=half))
    hk.mixed_precision.set_policy(Attention, jmp.Policy(param_dtype=full,
                                                        compute_dtype=full,
                                                        output_dtype=half))
    model_policy = jmp.Policy(param_dtype=full,
                              compute_dtype=half,
                              output_dtype=full)
    hk.mixed_precision.set_policy(Model, model_policy)

    @partial(jax.pmap, axis_name='device', donate_argnums=5)
    def train_step(x0: jnp.ndarray,
                   params_: optax.Params,
                   ema_: optax.Params,
                   opt_state_: optax.MultiStepsState,
                   rng: Any,
                   scale_: jmp.LossScale,
                   ) -> Tuple[optax.Params, optax.Params, optax.OptState,
                              jmp.LossScale, jnp.ndarray]:
        rng, rng_split = jax.random.split(rng)
        noise = jax.random.normal(rng, shape=x0.shape)
        rng, rng_split = jax.random.split(rng)
        t = jax.random.uniform(rng_split, shape=(len(x0),))
        snr = cosine_snr(t)
        xt = sample_q(x0, noise, snr)
        gradients, (loss, scale_) = gradient_fn(params_, xt, snr, noise,
                                                scale_, rng)
        gradients = model_policy.cast_to_compute(gradients)
        gradients = scale.unscale(gradients)
        gradients = jax.lax.pmean(gradients, axis_name='device')
        gradients = model_policy.cast_to_param(gradients)
        updates, opt_state_ = opt.update(gradients, opt_state_, params=params_)
        params_ = optax.apply_updates(params_, updates)
        ema_ = optax.incremental_update(params_, ema_,
                                        step_size=1. - cfg.tr.ema_alpha)
        return params_, ema_, opt_state_, scale_, loss ** 0.5

    batch_it = (batch for _ in count()
                for batch in iter(dataset.dataloader()))
    device_count = jax.device_count()
    logger.info(f'Devices found: {device_count}.')
    # Broadcast the params, ema and opt_state to all devices so we can use them
    # inside train_step, which is compiled with jax.pmap. Only the gradients
    # will be shared between each device, minimizing the communication
    # overhead.
    arr_broadcast = lambda x: jnp.broadcast_to(x, (device_count, *x.shape))
    pytree_broadcast = lambda t: jax.tree_util.tree_map(arr_broadcast, t)
    p_params = pytree_broadcast(params)
    p_ema = pytree_broadcast(ema)
    p_opt_state = pytree_broadcast(opt_state)
    p_scale = pytree_broadcast(scale)
    # The inverse of broadcasting the params etc. across devices. This is used
    # when we want to yield the current training state to downstream functions,
    # e.g. autosave.
    pytree_collapse = lambda t: jax.tree_util.tree_map(lambda x: x[0], t)
    while True:
        maes = []
        for _ in range(cfg.tr.gradient_accumulation_steps):
            batch = next(batch_it)
            p_batch = rearrange(batch, '(d b) ... -> d b ...',
                                d=device_count)
            p_rng = jnp.array([next(rngs) for _ in range(device_count)])
            state = train_step(p_batch, p_params, p_ema, p_opt_state, p_rng,
                               p_scale)
            p_params, p_ema, p_opt_state, p_scale, p_mae = state
            maes.append(float(p_mae.mean()))
        ctx.iteration += 1
        if ctx.iteration % cfg.tr.yield_freq == 0:
            yield Pack(workdir=workdir,
                       cfg=cfg,
                       ctx=ctx,
                       params=pytree_collapse(p_params),
                       ema=pytree_collapse(p_ema),
                       opt_state=pytree_collapse(p_opt_state),
                       rngs=rngs,
                       mae=jnp.mean(jnp.array(maes)))


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
            maes_: jnp.ndarray = jnp.array(maes)
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
    run.log({'MAE': mean_mae}, step=pack.ctx.iteration)


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
