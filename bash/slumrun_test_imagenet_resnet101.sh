#!/bin/bash
#SBATCH --job-name=resnet101_imgnet
#SBATCH --output=resnet101_imgnet.out
#SBATCH --error=resnet101_imgnet.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=144:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=20G
#SBATCH --mail-type=R,END,FAIL # Send email on job END and FAIL
#SBATCH --mail-user=soroush1@yorku.ca

srun python /home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/scripts/imagenet_script_resnet101.py