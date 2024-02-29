#!/bin/bash
#SBATCH --job-name=vgg
#SBATCH --output=result.out
#SBATCH --error=error.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=40G
#SBATCH --mail-type=END,FAIL # Send email on job END and FAIL
#SBATCH --mail-user=soroush1@yorku.ca

srun python /home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/lamem_script.py