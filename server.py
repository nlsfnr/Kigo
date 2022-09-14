#!/usr/bin/env python3
from typing import Optional
from pathlib import Path
from functools import partial
from io import BytesIO
import base64
from PIL import Image
import click
from chex import Array
import haiku as hk
import jax
from flask import Flask, Response
import numpy as np

from kigo.utils import get_logger, get_rngs
from kigo import persistence
from kigo import nn
from kigo import diffusion
from kigo.configs import Config


logger = get_logger('kigo.server')


@click.command('Server')
@click.argument('checkpoint', type=persistence.get_checkpoint)
@click.option('--port', '-p', type=int, default=8080)
@click.option('--host', '-h', type=str, default='127.0.0.1')
@click.option('--seed', type=int, default=None)
def cli(checkpoint: Path, port: int, host: str, seed: Optional[int]) -> None:
    '''A server wrapping Kigo!'''
    rngs = get_rngs(seed)
    cfg = persistence.load_cfg(checkpoint)
    params = persistence.load_ema(checkpoint)
    sampler = diffusion.Sampler(params, partial(nn.Model.from_cfg, cfg))
    app = build_app(sampler, cfg, rngs)
    app.run(host=host, port=port)


def build_app(sampler: diffusion.Sampler,
              cfg: Config,
              rngs: hk.PRNGSequence) -> Flask:
    app = Flask(__name__)

    @app.route('/', methods=['GET'])
    def index() -> Response:
        xT = jax.random.normal(next(rngs), shape=(1, *cfg.img.shape))
        x0 = sampler.sample_p(xT, 64, next(rngs), 0.1, 0.995)
        return Response(to_html(x0))

    return app


def to_base64_img(xt: Array) -> str:
    xt_np = np.asarray(xt).squeeze(0) * 0.5 + 0.5
    img = Image.fromarray((255 * xt_np).astype(np.uint8))
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    img_byte = buffer.getvalue()
    img_str = base64.b64encode(img_byte).decode()
    return "data:image/png;base64," + img_str


def to_html(xt: Array) -> str:
    return f'<img src="{to_base64_img(xt)}"/>'


if __name__ == '__main__':
    cli()
