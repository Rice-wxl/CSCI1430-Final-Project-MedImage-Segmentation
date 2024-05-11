#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH -p gpu --gres=gpu:1
#SBATCH --constraint=ampere
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -J run_vaih
#SBATCH -o run_vaih_%j.out
#SBATCH -e run_vaih_%j.err

# module load miniconda3/23.11.0s
# source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh 
# conda activate segdiff
# module load cuda

# python image_train_diff_vaih.py --lr 0.0001 --batch_size 4 --dropout 0.1 --rrdb_blocks 6 --diffusion_steps 100
# CUDA_VISIBLE_DEVICES=0,1 mpiexec -n 4 python image_train_diff_vaih.py --lr 0.0001 --batch_size 4 --dropout 0.1 --rrdb_blocks 6 --diffusion_steps 100
# CUDA_VISIBLE_DEVICES=0,1 srun --pmix=mpi -n 4 python image_train_diff_vaih.py --lr 0.0001 --batch_size 4 --dropout 0.1 --rrdb_blocks 6 --diffusion_steps 100

export PYTHONUSERBASE=$PWD/pyenv
export APPTAINER_CACHEDIR=/tmp
export APPTAINER_TMPDIR=/tmp
export APPTAINER_BINDPATH="/oscar/home/$USER,/oscar/scratch/$USER,/oscar/data"
export PATH=$PATH:$PWD/pyenv/bin
export export LD_LIBRARY_PATH=$PWD/pyenv/lib

<<<<<<< HEAD
CUDA_VISIBLE_DEVICES=0,1 srun apptainer exec --nv /oscar/runtime/software/external/ngc-containers/pytorch.d/x86_64.d/pytorch-24.03-py3 python train_brats.py --lr 0.0001 --batch_size 4 --dropout 0.1 --rrdb_blocks 6 --diffusion_steps 100
=======
CUDA_VISIBLE_DEVICES=0,1 mpiexec -n 1 srun apptainer exec --nv /oscar/runtime/software/external/ngc-containers/pytorch.d/x86_64.d/pytorch-24.03-py3 python image_train_diff_vaih.py --lr 0.0001 --batch_size 4 --dropout 0.1 --rrdb_blocks 6 --diffusion_steps 100
>>>>>>> 4cac8e4ed6e010574169d4bc2b839046615832ca
