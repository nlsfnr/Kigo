'''The implementation of the diffusion method.'''
from typing import Tuple, Any, Union, TypeVar
from chex import Array
import jax
import jax.numpy as jnp
from einops import rearrange

from .nn import ForwardFn


NumT = TypeVar('NumT', bound=Union[float, Array])


def expand(x: Union[float, Array], ref: Array) -> Array:
    if isinstance(x, float):
        x = jnp.array([x] * ref.shape[0])
    if isinstance(x, Array) and len(x.shape) == 1:
        x = rearrange(x, 'b -> b 1 1 1')
    assert isinstance(x, Array)
    return x


def gt0(x: NumT, eps: float = 1e-8) -> NumT:
    '''Ensures that x is greater than zero, i.e. can be safely used as a
    divisor or for sqrts.'''
    return jnp.clip(x, eps)


def cosine_snr(t: Union[float, Array], s: float = 0.008) -> Array:
    '''Signal-to-noise ratio according to a cosine schedule.'''
    t = jnp.array(t)
    t = jnp.clip(t, 0., 1.)
    return jnp.cos((t + s) / (1. + s) * jnp.pi / 2) ** 2


def sample_q(x0: Array, noise: Array, snr: Array) -> Array:
    snr = rearrange(snr, 'b -> b 1 1 1')
    # Eq. 4 in DDIM
    return gt0(snr) ** 0.5 * x0 + gt0(1. - snr) ** 0.5 * noise


def sample_p_step(xt: Array,
                  noise_pred: Array,
                  snr: Union[float, Array],
                  snr_next: Union[float, Array],
                  eta: Union[float, Array],
                  noise: Array,
                  clip_percentile: float = 0.995,
                  ) -> Array:
    snr = expand(snr, xt)
    snr_next = expand(snr_next, xt)
    eta = expand(eta, xt)
    # Eq. 16 in DDIM, we can interpolate between DDPM (when eta = 1) and DDIM
    # (when eta = 0).
    sigma = (eta
             * gt0((1 - snr_next) / gt0(1 - snr)) ** 0.5
             * gt0((1 - snr) / gt0(snr_next)) ** 0.5)
    # Eq. 9 in DDIM
    x0_hat = (xt - gt0(1. - snr) ** 0.5 * noise_pred) / gt0(snr) ** 0.5
    # Dynamic thresholding from Imagen by the Google Brain Team.
    s = jnp.quantile(jnp.abs(x0_hat), clip_percentile, axis=(1, 2, 3),
                     keepdims=True)
    x0_hat = jnp.where(s > 1.,
                       jnp.clip(x0_hat, -s, s) / gt0(s),
                       x0_hat)
    # Eq. 12 in DDIM
    xt = (x0_hat * gt0(snr_next) ** 0.5
          + noise_pred * gt0(1. - snr_next - sigma ** 2) ** 0.5
          + noise * sigma)
    return xt


def sample_p(xT: Array,
             forward_fn: ForwardFn,
             steps: int,
             rng: Any,
             eta: Union[float, Array] = 0.,
             clip_percentile: float = 0.995,
             ) -> Array:

    def body_fn(index: int, state: Tuple[Array, Any]) -> Tuple[Array, Any]:
        xt, rng = state
        rng, rng_split = jax.random.split(rng)
        snr = jnp.repeat(cosine_snr(1. - index / steps), len(xt))
        snr_next = jnp.repeat(cosine_snr(1. - (index + 1) / steps), len(xt))
        noise_pred = forward_fn(xt, snr, False)
        noise = jax.random.normal(rng_split, shape=xt.shape)
        xt_next = sample_p_step(xt, noise_pred, snr, snr_next, eta, noise,
                                clip_percentile)
        return xt_next, rng

    initial_state = xT, rng
    x0, _ = jax.lax.fori_loop(0, steps, body_fn, initial_state)
    return x0
