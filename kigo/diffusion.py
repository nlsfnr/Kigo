'''The implementation of the diffusion method.'''
from typing import Tuple, Any, Union, TypeVar
import jax
import jax.numpy as jnp
from einops import rearrange

from .nn import ForwardFn


NumT = TypeVar('NumT', bound=Union[float, jnp.ndarray])


def gt0(x: NumT, eps: float = 1e-12) -> NumT:
    '''Ensures that x is greater than zero, i.e. can be safely used as a
    divisor or for sqrts.'''
    return jnp.clip(x, eps)


def cosine_snr(t: Union[float, jnp.ndarray], s: float = 0.008) -> jnp.ndarray:
    '''Signal-to-noise ratio according to a cosine schedule.'''
    t = jnp.array(t)  # type: ignore
    t = jnp.clip(t, 0., 1.)
    return jnp.cos((t + s) / (1. + s) * jnp.pi / 2) ** 2


def sample_q(x0: jnp.ndarray,
             noise: jnp.ndarray,
             snr: jnp.ndarray,
             ) -> jnp.ndarray:
    snr = rearrange(snr, 'b -> b 1 1 1')
    # Eq. 4 in DDIM
    return gt0(snr) ** 0.5 * x0 + gt0(1. - snr) ** 0.5 * noise


def predict_x0(xt: jnp.ndarray,
               noise_pred: jnp.ndarray,
               snr: Union[float, jnp.ndarray],
               clip_percentile: float = 0.995,
               ) -> jnp.ndarray:
    # Eq. 9 in DDIM
    x0_hat = (xt - gt0(1. - snr) ** 0.5 * noise_pred) / gt0(snr ** 0.5)
    # Dynamic thresholding from Imagen by the Google Brain Team.
    s = jnp.quantile(jnp.abs(x0_hat), clip_percentile, axis=(1, 2, 3),
                     keepdims=True)
    xt = jnp.where(s > 1., jnp.clip(x0_hat, -s, s) / gt0(s), x0_hat)
    assert isinstance(xt, jnp.ndarray)
    return xt


def sample_p_step(xt: jnp.ndarray,
                  noise_pred: jnp.ndarray,
                  snr: Union[float, jnp.ndarray],
                  snr_next: Union[float, jnp.ndarray],
                  eta: Union[float, jnp.ndarray],
                  noise: jnp.ndarray,
                  clip_percentile: float = 0.995,
                  ) -> jnp.ndarray:
    # Eq. 16 in DDIM, we can interpolate between DDPM (when eta = 1) and DDIM
    # (when eta = 0).
    sigma = (eta
             * gt0((1 - snr_next) / gt0(1 - snr)) ** 0.5
             * gt0((1 - snr) / gt0(snr_next)) ** 0.5)
    # Eq. 12 in DDIM
    x0_hat = predict_x0(xt, noise_pred, snr, clip_percentile)
    xt = (x0_hat * gt0(snr_next) ** 0.5
          + noise_pred * gt0(1. - snr_next - sigma ** 2) ** 0.5
          + noise * sigma)
    return xt


def sample_p(xT: jnp.ndarray,
             forward_fn: ForwardFn,
             steps: int,
             rng: Any,
             eta: Union[float, jnp.ndarray] = 0.,
             clip_percentile: float = 0.995,
             ) -> jnp.ndarray:

    def body_fn(index: int, state: Tuple[jnp.ndarray, Any]
                ) -> Tuple[jnp.ndarray, Any]:
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
