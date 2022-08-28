# Kigo

*Kigo* (Jap., "season word") is a word or phrase associated with a particular
season, used in traditional forms of Japanese poetry. Kigos commonly occur in
Haikus.

## Setup

Because each JAX installation is different depending on your CUDA version, Jax
is not listed as a requirement in requirements.txt and thus needs to be manually
installed.

Kigo was developed using `pip install "jax[cuda11_cudnn805]" -f
https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`.

To install all other dependencies, run either `make` or `pip3 install -r
requirements.txt`. The former will also do type-checking, linting and
unit-testing.

## Usage

All interactions with Kigo are done through the `./cli.py`. Run `./cli.py
--help` for more information.

A typical workflow to create, train and then sample from a model could look like
this:

```bash
# Create and save a new model to `zoo/my-model`, including its EMA, optimizer
# and a copy of the config from `configs/my-config.yaml`.
./cli.py init zoo/my-model/ configs/my-config.yaml

# Train the model until interrupted with e.g. Ctrl+c.
./cli.py train zoo/my-model/

# Sample from the model, with 64 steps, mixing DDIM and DDPM (ema=0.5) and
# saving the result to my-image.png.
./cli.py syn zoo/my-model/ --steps 64 --ema 0.5 --out my-image.png
```

The configuration specified in the `./cli.py init` command should be
self-explanatory. A copy of it is stored with the initial and all later
checkpoints stored inside the working directory, which is `zoo/my-model/` in
this case. Using symlinks, Kigo will automatically find the latest checkpoint
inside of a working directory, however, one can sample from a specific
checkpoint by running e.g. `./cli.py syn zoo/my-model/001000/`.
