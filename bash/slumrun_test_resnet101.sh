#!/bin/bash
#SBATCH --job-name=resnet101
#SBATCH --output=result_resnet101.out
#SBATCH --error=error_resnet101.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --mail-type=R,END,FAIL # Send email on job END and FAIL
#SBATCH --mail-user=soroush1@yorku.ca

srun python /home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/lamem_script_resnet101.py