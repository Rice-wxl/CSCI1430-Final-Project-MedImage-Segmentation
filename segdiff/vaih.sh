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

# CUDA_VISIBLE_DEVICES=0,1 srun apptainer exec --nv /oscar/runtime/software/external/ngc-containers/pytorch.d/x86_64.d/pytorch-24.03-py3 python image_train_diff_vaih.py --lr 0.0001 --batch_size 4 --dropout 0.1 --rrdb_blocks 6 --diffusion_steps 100
# CUDA_VISIBLE_DEVICES=0 srun apptainer exec --nv /oscar/runtime/software/external/ngc-containers/pytorch.d/x86_64.d/pytorch-24.03-py3 python image_sample_diff_vaih.py --model_path /users/xwang259/CSCI1430-Final-Project-MedImage-Segmentation/CSCI1430-Final-Project-MedImage-Segmentation/logs/2024-05-13-00-11-53-894787_vaih_256_6_0.0001_4_100_0.1_0/model_1000.pt

CUDA_VISIBLE_DEVICES=0,1 srun apptainer exec --nv /oscar/runtime/software/external/ngc-containers/pytorch.d/x86_64.d/pytorch-24.03-py3 python train_isic.py --lr 0.0001 --batch_size 4 --dropout 0.1 --rrdb_blocks 6 --diffusion_steps 100

