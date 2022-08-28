from typing import Iterator, NamedTuple, Any, Optional, Tuple
from dataclasses import dataclass
from functools import partial
from itertools import count
import haiku as hk
import optax
import jax
import jax.numpy as jnp
from einops import rearrange

from .diffusion import sample_q, cosine_snr
from .utils import Directory
from .configs import Config
from .data import Dataset
from .nn import Model
from . import persistence


@dataclass
class Context:
    iteration: int = 0

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

    @partial(jax.pmap, axis_name='device', donate_argnums=5)
    def train_step(x0: jnp.ndarray,
                   params_: optax.Params,
                   ema_: optax.Params,
                   opt_state_: optax.MultiStepsState,
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
        gradients = jax.lax.pmean(gradients, axis_name='device')
        mae = loss ** 0.5
        updates, opt_state_ = opt.update(gradients, opt_state_, params=params_)
        params_ = optax.apply_updates(params_, updates)
        ema_ = optax.incremental_update(params_, ema_,
                                        step_size=1. - cfg.tr.ema_alpha)
        return params_, ema_, opt_state_, mae

    batch_it = (batch for _ in count()
                for batch in iter(dataset.dataloader()))
    device_count = jax.device_count()
    # Broadcast the params, ema and opt_state to all devices so we can use them
    # inside train_step, which is compiled with jax.pmap. Only the gradients
    # will be shared between each device, minimizing the communication
    # overhead.
    arr_broadcast = lambda x: jnp.broadcast_to(x, (device_count, *x.shape))
    pytree_broadcast = lambda t: jax.tree_util.tree_map(arr_broadcast, t)
    p_params = pytree_broadcast(params)
    p_ema = pytree_broadcast(ema)
    p_opt_state = pytree_broadcast(opt_state)
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
            state = train_step(p_batch, p_params, p_ema, p_opt_state, p_rng)
            p_params, p_ema, p_opt_state, p_mae = state
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
