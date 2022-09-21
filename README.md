# Kigo

*Kigo* (Jap., "season word") is a word or phrase associated with a particular
season, used in traditional forms of Japanese poetry. Kigos commonly occur in
Haikus.

Kigo implements a diffusion model using Jax, DM-Haiku, Optax and other tools
from the DeepMind stack.

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

## Code

This section is supposed to give a good overview of the code and its structure.
To this end, some interesting parts of the code are given. All of the code is
annotated with types, type-checked with MyPy and linted with Flake8.

### Model

The implementation of the model can be found
[here](https://github.com/nlsfnr/Kigo/blob/master/kigo/nn.py#L191). The model is
a version of UNet with a recursive implementation. It is instantiated using a
`Config`.

### Config

To serve as a single source of truth and make model versioning and persistence
easier, a configuration is loaded from a `.yaml` file each time that a model is
interacted with. The structure of the configuration is defined
[here](https://github.com/nlsfnr/Kigo/blob/master/kigo/configs.py).

### Persistence

To store and load checkpoints including optimizer, model EMA, configuration and
others, the `persistence` is used. It is implemented
[here](https://github.com/nlsfnr/Kigo/blob/master/kigo/persistence.py). It also
provides functionality to automatically find and define checkpoint directories.

### Training

The training loop and other associated functionality is implemented
[here](https://github.com/nlsfnr/Kigo/blob/master/kigo/training.py). The
[`train`
function](https://github.com/nlsfnr/Kigo/blob/master/kigo/training.py#L117)
returns an iterator of
[`Pack`s](https://github.com/nlsfnr/Kigo/blob/master/kigo/training.py#L45) which
contain e.g. the current model parameter, the optimizer state etc. Said iterator
is then passed to functions downstream, such as
[logging](https://github.com/nlsfnr/Kigo/blob/master/kigo/training.py#L177),
[automatic
saving](https://github.com/nlsfnr/Kigo/blob/master/kigo/training.py#L153) and
[remote
logging](https://github.com/nlsfnr/Kigo/blob/master/kigo/training.py#L185) on
[Weights and Biases](https://wandb.ai). This allows for maximum flexibility and
keeps each function clean and understandable.

The train function acts as a wrapper around the [train_step
function](https://github.com/nlsfnr/Kigo/blob/master/kigo/training.py#L92). The
latter is `jax.pmap`ped across all devices, allowing for multi-device training.

### Diffusion

All diffusion specific algorithms are implemented
[here](https://github.com/nlsfnr/Kigo/blob/master/kigo/diffusion.py). The meat
of the sampling process is implemented inside
[sample_p_step](https://github.com/nlsfnr/Kigo/blob/master/kigo/diffusion.py#L45)
which performs one iteration of DDIM sampling given the current image, the
model's prediction of the noise the signal-to-noise-ratio (SNR) and other scalar
parameters. It uses the dynamic thresholding from Google Brain's Imagen.

## Other

This project was implemented using tools from the [DeepMind Jax
ecosystem](https://www.deepmind.com/blog/using-jax-to-accelerate-our-research).
