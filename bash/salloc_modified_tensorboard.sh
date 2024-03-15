#!/bin/bash
#SBATCH --job-name=tensorboard
#SBATCH --output=tensorboard.out
#SBATCH --error=tensorboard.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --mem=90G
#SBATCH --mail-type=R,END,FAIL # Send email on job END and FAIL
#SBATCH --mail-user=soroush1@yorku.ca


srun ./bash/tb.sh /home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/runs/imagenet
