#!/bin/bash
#SBATCH --job-name=prepare_imagenetDataset
#SBATCH --output=result_imagenet_prepration.out
#SBATCH --error=error_imagenet_prepration.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --time=01:00:00
#SBATCH --mem=8GB
#SBATCH --mem=15GB
#SBATCH --mail-type=READY,END,FAIL # Send email on job END and FAIL
#SBATCH --mail-user=soroush1@yorku.ca

srun python /home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/scripts/imagenet_script.py




