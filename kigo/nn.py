'''Implementation of the model and modules.'''
from __future__ import annotations
from typing import Optional, List, Callable, Tuple
from functools import partial
import jax
import jax.numpy as jnp
import haiku as hk
from einops import rearrange

from .configs import ModelConfig, UBlockConfig, Config


class SinusoidalEmbedding:

    def __init__(self, width: int) -> None:
        assert width % 2 == 0, 'Expected even width for timestep embeddings'
        self.width = width
        self.freqs = jnp.logspace(0, 1., width // 2)

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        xs = self.freqs[None, :] * t[:, None]
        return jnp.concatenate([jnp.sin(xs), jnp.cos(xs)], axis=-1)


class Attention(hk.Module):

    def __init__(self,
                 channels: int,
                 heads: int,
                 head_channels: int,
                 groupnorm_groups: int,
                 name: Optional[str] = None
                 ) -> None:
        super().__init__(name)
        self.scale = head_channels ** -0.5
        hidden_channels = head_channels * heads
        self.heads = heads
        self.norm = hk.GroupNorm(groupnorm_groups)
        self.to_qkv = hk.Conv2D(hidden_channels * 3, 1, with_bias=False)
        self.out_proj = hk.Conv2D(channels, 1,
                                  w_init=hk.initializers.Constant(0.),
                                  b_init=hk.initializers.Constant(0.))

    def __call__(self, xt: jnp.ndarray) -> jnp.ndarray:
        _, h, w, _ = xt.shape
        xt_norm = self.norm(xt)
        xt_norm = jax.nn.silu(xt_norm)
        # Compute queries, keys and values
        qkv = jnp.split(self.to_qkv(xt_norm), 3, axis=-1)
        q, k, v = [rearrange(arr, "b x y (h c) -> b h c (x y)", h=self.heads)
                   for arr in qkv]
        q = q * self.scale
        # Match queries to keys
        pattern = "b h d i, b h d j -> b h i j"
        sim: jnp.ndarray = jnp.einsum(pattern, q, k)  # type: ignore
        sim = sim - jnp.amax(sim, axis=-1, keepdims=True)
        attn: jnp.ndarray = jax.nn.softmax(sim, axis=-1)
        # Compute the values for each head
        pattern = "b h i j, b h d j -> b h i d"
        out = jnp.einsum(pattern, attn, v)  # type: ignore
        out = rearrange(out, "b h (x y) d -> b x y (h d)", x=h, y=w)
        out = self.out_proj(out)
        return out + xt


class ResBlock(hk.Module):

    def __init__(self,
                 channels: int,
                 groupnorm_groups: int,
                 att_heads: int,
                 att_head_channels: int,
                 dropout: float,
                 name: Optional[str] = None,
                 ) -> None:
        super().__init__(name)
        self.dropout = dropout
        self.norm_1 = hk.GroupNorm(groupnorm_groups)
        self.conv_1 = hk.Conv2D(channels, 3)
        self.norm_2 = hk.GroupNorm(groupnorm_groups)
        self.affine = hk.Linear(channels * 2,
                                w_init=hk.initializers.Constant(0.),
                                b_init=hk.initializers.Constant(0.))
        self.conv_2 = hk.Conv2D(channels, 3,
                                w_init=hk.initializers.Constant(0.),
                                b_init=hk.initializers.Constant(0.))
        self.residual_conv = hk.Conv2D(channels, 1)
        self.attention = (Attention(channels, att_heads, att_head_channels,
                                    groupnorm_groups)
                          if att_heads > 0 else
                          None)

    def __call__(self, xt: jnp.ndarray, snr_emb: jnp.ndarray, training: bool
                 ) -> jnp.ndarray:
        h = self.norm_1(xt)
        h = jax.nn.silu(h)
        h = self.conv_1(h)
        h = self.norm_2(h)
        wb = self.affine(snr_emb)
        w, b = jnp.split(wb, 2, axis=-1)
        w = rearrange(w, 'b c -> b 1 1 c')
        b = rearrange(b, 'b c -> b 1 1 c')
        h = h * (1. + w) + b
        if training:
            h = hk.dropout(hk.next_rng_key(), self.dropout, h)
        h = jax.nn.silu(h)
        h = self.conv_2(h)
        h = h + self.residual_conv(xt)
        if self.attention is not None:
            h = self.attention(h)
        return h


class UBlock(hk.Module):

    def __init__(self,
                 ub_cfgs: List[UBlockConfig],
                 outer_channels: int,
                 is_outer: bool = False,
                 name: Optional[str] = None,
                 ) -> None:
        super().__init__(name)
        self.ub_cfg, *inner_ub_cfgs = ub_cfgs
        ch = self.ub_cfg.channels
        self.in_proj = hk.Sequential([
            hk.GroupNorm(self.ub_cfg.groupnorm_groups),
            jax.nn.silu,
            (hk.Conv2D(ch, 1)
             if is_outer else
             hk.Conv2D(ch, 3, stride=2)),
        ])
        self.down_blocks = [
            ResBlock(ch,
                     self.ub_cfg.groupnorm_groups,
                     self.ub_cfg.attention_heads,
                     self.ub_cfg.attention_head_channels,
                     self.ub_cfg.dropout)
            for _ in range(self.ub_cfg.blocks)
        ]
        if inner_ub_cfgs:
            self.inner_ublock = UBlock(inner_ub_cfgs, outer_channels=ch)
            self.concat_proj = hk.Conv2D(ch, 1)
        else:
            self.inner_ublock = None
            self.concat_proj = None
        self.up_blocks = [
            ResBlock(ch,
                     self.ub_cfg.groupnorm_groups,
                     self.ub_cfg.attention_heads,
                     self.ub_cfg.attention_head_channels,
                     self.ub_cfg.dropout)
            for _ in range(self.ub_cfg.blocks)
        ]
        self.out_proj = hk.Sequential([
            hk.GroupNorm(self.ub_cfg.groupnorm_groups),
            jax.nn.silu,
            (hk.Conv2D(outer_channels, 1)
             if is_outer else
             hk.Conv2DTranspose(outer_channels, 3, stride=2)),
        ])

    def __call__(self,
                 xt: jnp.ndarray,
                 snr_emb: jnp.ndarray,
                 training: bool
                 ) -> jnp.ndarray:
        h = self.in_proj(xt)
        for block in self.down_blocks:
            h = block(h, snr_emb, training)
        if self.inner_ublock is not None:
            assert self.concat_proj is not None
            i = self.inner_ublock(h, snr_emb, training)
            h = jnp.concatenate([h, i], axis=-1)
            h = self.concat_proj(h)
        for block in self.up_blocks:
            h = block(h, snr_emb, training)
        return self.out_proj(h)


class Model(hk.Module):

    def __init__(self,
                 model_cfg: ModelConfig,
                 name: Optional[str] = None,
                 ) -> None:
        super().__init__(name=name)
        self.model_cfg = model_cfg
        oc = self.model_cfg.outer_channels
        self.in_proj = hk.Conv2D(oc, 3)
        self.ublock = UBlock(self.model_cfg.blocks,
                             outer_channels=self.model_cfg.outer_channels,
                             is_outer=True)
        self.out_proj = hk.Sequential([
            hk.GroupNorm(self.model_cfg.outer_groupnorm_groups),
            jax.nn.silu,
            hk.Conv2D(self.model_cfg.output_channels, 3,
                      w_init=hk.initializers.Constant(0.),
                      b_init=hk.initializers.Constant(0.)),
        ])
        self.snr_mlp = hk.Sequential([
            SinusoidalEmbedding(self.model_cfg.snr_sinusoidal_embedding_width),
            jax.nn.silu,
            hk.Linear(self.model_cfg.snr_embedding_width),
            jax.nn.silu,
        ])

    @classmethod
    def from_cfg(cls, cfg: Config, name: Optional[str] = None) -> Model:
        '''Constructs a model from the given configuration.'''
        return cls(cfg.model, name=name)

    def __call__(self, xt: jnp.ndarray, snr: jnp.ndarray, training: bool
                 ) -> jnp.ndarray:
        snr_emb = self.snr_mlp(snr)
        h = self.in_proj(xt)
        h = self.ublock(h, snr_emb, training)
        h = self.out_proj(h)
        return h


ForwardFn = Callable[[jnp.ndarray, jnp.ndarray, bool], jnp.ndarray]


def get_params_and_forward_fn(cfg: Config,
                              rng_key: hk.PRNGSequence,
                              params: Optional[hk.Params] = None,
                              ) -> Tuple[hk.Params, ForwardFn]:

    def forward_fn(xt: jnp.ndarray,
                   snr: jnp.ndarray,
                   training: bool = False
                   ) -> jnp.ndarray:
        model = Model.from_cfg(cfg)
        return model(xt, snr, training)

    S, B = cfg.img.size, 1
    repr_xt = jnp.zeros((B, S, S, cfg.model.input_channels))  # type: ignore
    repr_t = jnp.zeros((B,))  # type: ignore
    model = hk.transform(forward_fn)
    if params is None:
        params = model.init(next(rng_key), repr_xt, repr_t)
    return params, partial(model.apply, params, next(rng_key))
