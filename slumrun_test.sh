#!/bin/bash
#SBATCH --job-name=my_training_job
#SBATCH --output=result.out
#SBATCH --error=error.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

srun python main.py