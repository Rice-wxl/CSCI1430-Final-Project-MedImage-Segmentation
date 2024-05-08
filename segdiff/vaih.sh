#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH -p gpu --gres=gpu:1 
#SBATCH -n 8
#SBATCH --mem=16G
#SBATCH -J run_vaih
#SBATCH -o run_vaih_%j.out
#SBATCH -e run_vaih_%j.err

module load cuda
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh 

conda activate segdiff

python3 BLIP_patching.py --samples full --block_name text_encoder --kind attention_block