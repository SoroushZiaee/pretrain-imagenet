#!/bin/bash
#SBATCH --job-name=vgg16_imgnet
#SBATCH --output=vgg16_imgnet.out
#SBATCH --error=vgg16_imgnet.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=20G
#SBATCH --mail-type=R,END,FAIL # Send email on job END and FAIL
#SBATCH --mail-user=soroush1@yorku.ca

srun python /home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/scripts/imagenet_script_vgg16.py