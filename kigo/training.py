from typing import Iterator, NamedTuple, Any, Optional, Tuple
from dataclasses import dataclass
from itertools import count
import haiku as hk
import optax
import jax
import jax.numpy as jnp

from .diffusion import sample_q, cosine_snr
from .utils import Directory
from .configs import Config
from .data import Dataset
from .nn import Model
from . import persistence


@dataclass
class Context:
    iteration: int = 0

    def periodically(self, freq: int, skip_first: bool = False) -> bool:
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


class TrainState(NamedTuple):
    params: optax.Params
    ema: optax.Params
    opt_state: optax.MultiStepsState
    rng: Any


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

    def forward_fn(x0: jnp.ndarray,
                   snr: jnp.ndarray,
                   ) -> jnp.ndarray:
        return Model.from_cfg(cfg)(x0, snr, False)

    forward = hk.transform(forward_fn)

    def loss_fn(params_: optax.Params,
                xt: jnp.ndarray,
                snr: jnp.ndarray,
                noise: jnp.ndarray,
                rng: Any,
                ) -> jnp.ndarray:
        noise_pred = forward.apply(params_, rng, xt, snr)
        return jnp.mean((noise_pred - noise) ** 2)

    gradient_fn = jax.value_and_grad(loss_fn)
    opt, opt_state = get_opt_and_opt_state(cfg, params, opt_state)

    @jax.jit
    def train_step(params_: optax.Params,
                   ema_: optax.Params,
                   opt_state_: optax.MultiStepsState,
                   x0: jnp.ndarray,
                   rng: Any,
                   ) -> Tuple[optax.Params, optax.Params, optax.OptState,
                              jnp.ndarray]:
        rng, rng_split = jax.random.split(rng)
        noise = jax.random.normal(rng, shape=x0.shape)
        rng, rng_split = jax.random.split(rng)
        t = jax.random.uniform(rng_split, shape=(len(x0),))
        snr = cosine_snr(t)
        xt = sample_q(x0, noise, snr)
        loss, gradients = gradient_fn(params_, xt, snr, noise, rng)
        mae = loss ** 0.5
        updates, opt_state_ = opt.update(gradients, opt_state_, params=params_)
        params_ = optax.apply_updates(params_, updates)
        ema_ = optax.incremental_update(params_, ema_,
                                        step_size=1. - cfg.tr.ema_alpha)
        return params_, ema_, opt_state_, mae

    batch_it = (batch for _ in count()
                for batch in iter(dataset.dataloader()))
    while True:
        mae_acc = 0.
        for _ in range(cfg.tr.yield_freq):
            for _ in range(cfg.tr.gradient_accumulation_steps):
                params, ema, opt_state, mae = train_step(params,
                                                         ema,
                                                         opt_state,
                                                         next(batch_it),
                                                         next(rngs))
                mae_acc += float(mae) / cfg.tr.yield_freq
            ctx.iteration += 1
        yield Pack(workdir=workdir, cfg=cfg, ctx=ctx, params=params, ema=ema,
                   opt_state=opt_state, rngs=rngs,
                   mae=mae_acc / cfg.tr.gradient_accumulation_steps)


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
            if pack.ctx.periodically(pack.cfg.tr.save_checkpoint_freq,
                                     skip_first=True):
                cp = pack.workdir / str(pack.ctx.iteration).zfill(6)
                _save_pack(cp, pack)
            if pack.ctx.periodically(pack.cfg.tr.save_freq,
                                     skip_first=True):
                _save_pack(pack.workdir / 'latest', pack)
            yield pack
    finally:
        if pack is not None:
            _save_pack(pack.workdir / 'latest', pack)
