#!/bin/bash
#SBATCH --job-name=cs1430_MedT
#SBATCH --output=log_%J.log
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --nodes=1	
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128gb
#SBATCH --time=1-00:00:00   
date;hostname;pwd

export PYTHONUSERBASE=$PWD/cs1430
export APPTAINER_CACHEDIR=/tmp
export APPTAINER_TMPDIR=/tmp
export APPTAINER_BINDPATH="/oscar/home/$USER,/oscar/scratch/$USER,/oscar/data"
export PATH=$PATH:$PWD/cs1430/bin

srun apptainer exec --nv /oscar/runtime/software/external/ngc-containers/pytorch.d/x86_64.d/pytorch-24.03-py3 \
    python train.py \
    --direc "./results/" \
    --batch_size 4 \
    --modelname "MedT" \
    --epoch 400 \
    --save_freq 50 \
    --learning_rate 0.0001 \
    --imgsize 240

