#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH -p gpu --gres=gpu:1 
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -J run_vaih
#SBATCH -o run_vaih_%j.out

module load cuda
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh 

conda activate segdiff

python image_train_diff_vaih.py --lr 0.0001 --batch_size 4 --dropout 0.1 --rrdb_blocks 6 --diffusion_steps 100

# CUDA_VISIBLE_DEVICES=0,1 mpiexec -n 4 python image_train_diff_vaih.py --lr 0.0001 --batch_size 4 --dropout 0.1 --rrdb_blocks 6 --diffusion_steps 100
