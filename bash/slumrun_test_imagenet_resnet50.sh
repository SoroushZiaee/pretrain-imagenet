#!/bin/bash
#SBATCH --job-name=resnet50_imgnet
#SBATCH --output=resnet50_imgnet.out
#SBATCH --error=resnet50_imgnet.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --time=90:00:00
#SBATCH --gres=gpu:t4:4
#SBATCH --mem=20G
#SBATCH --mail-type=R,END,FAIL # Send email on job END and FAIL
#SBATCH --mail-user=soroush1@yorku.ca

srun python /home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/scripts/imagenet_script_resnet50.py