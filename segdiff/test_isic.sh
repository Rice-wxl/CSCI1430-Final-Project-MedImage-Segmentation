#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH -p gpu --gres=gpu:1
#SBATCH --constraint=ampere
#SBATCH -n 1
#SBATCH --mem=128G
#SBATCH -J run_vaih
#SBATCH -o run_vaih_%j.out
#SBATCH -e run_vaih_%j.err

export PYTHONUSERBASE=$PWD/pyenv
export APPTAINER_CACHEDIR=/tmp
export APPTAINER_TMPDIR=/tmp
export APPTAINER_BINDPATH="/oscar/home/$USER,/oscar/scratch/$USER,/oscar/data"
export PATH=$PATH:$PWD/pyenv/bin
export export LD_LIBRARY_PATH=$PWD/pyenv/lib

CUDA_VISIBLE_DEVICES=0 srun apptainer exec --nv /oscar/runtime/software/external/ngc-containers/pytorch.d/x86_64.d/pytorch-24.03-py3 python val_isic.py  --model_path ../logs/2024-05-13-01-23-09-042170_vaih_256_6_0.0001_4_100_0.1_0/model_5000.pt
