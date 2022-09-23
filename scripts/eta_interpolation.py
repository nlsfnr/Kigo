#!/usr/bin/env python3
from math import log10, ceil
from subprocess import run
from pathlib import Path
import click
import numpy as np

import sys
sys.path.append('..')

from kigo import persistence


@click.command
@click.argument('checkpoint', type=persistence.get_checkpoint)
@click.option('--low', '-l', type=float, default=0.)
@click.option('--high', '-h', type=float, default=1.)
@click.option('--num', '-n', type=int, default=16)
@click.option('--out', '-o', type=Path, default=Path('./eta-interpolations/'))
@click.option('--seed', '-s', type=int, default=1)
def main(checkpoint: Path,
         low: float,
         high: float,
         num: int,
         out: Path,
         seed: int,
         ) -> None:
    assert 0. <= low <= 1.
    assert 0. <= high <= 1.
    assert 0 < num
    out.mkdir(parents=True, exist_ok=True)
    low = low ** 2
    for i in np.linspace(0, 1., num=num):
        eta = low + (high - low) * i ** 2
        run(['../cli.py',
             'syn',
             str(checkpoint),
             '--eta', str(eta),
             '--seed', str(seed),
             '--out', str(out / f'{eta}.png')])


if __name__ == '__main__':
    main()
