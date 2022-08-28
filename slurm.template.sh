#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=%(cpus)d
#SBATCH --mem=%(memory)dgb
#SBATCH --output=%(logfile)s
#SBATCH --partition=gpu
#SBATCH --gres=gpu:%(gpus)d
#SBATCH --time=%(time)s
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=%(email)s

# Script to be run on compute clusters using Slurm.
# The Slurm parameters are set by the CLI. See `./cli.py slurm --help`.

echo "Loading CUDA 11.4.1 with cuDNN 9.2.1"
module load numlib/cuDNN/8.2.1.32-CUDA-11.3.1

pushd %(kigo_dir)s
[ -d .venv/ ] && . .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
python3 -m pip install --upgrade -r requirements.txt
./cli.py train %(checkpoint)s
popd
