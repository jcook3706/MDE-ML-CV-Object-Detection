#!/bin/bash

# Scheduler options
#SBATCH -N1
#SBATCH --ntasks-per-node=16
#SBATCH -t 14:00:00
#SBATCH -p v100_normal_q
#SBATCH -A creed_jones
#SBATCH --gres=gpu:1

# setup environment
module load Anaconda3/2020.11
source activate py38condaenv
source ~/py38venv/bin/activate

# run executable
python train.py
