#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH -p gpu --gres=gpu:1 
#SBATCH -n 8
#SBATCH --mem=16G
#SBATCH -J run_vaih
#SBATCH -o run_vaih_%j.out
#SBATCH -e run_vaih_%j.err

module load python/3.9.16s-x3wdtvt
source /users/mgolovan/data/mgolovan/llava/bin/activate

python3 BLIP_patching.py --samples full --block_name text_encoder --kind attention_block