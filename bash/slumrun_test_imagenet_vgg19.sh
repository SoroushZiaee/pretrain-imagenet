#!/bin/bash
#SBATCH --job-name=vgg19_imgnet
#SBATCH --output=vgg19_imgnet.out
#SBATCH --error=vgg19_imgnet.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=144:00:00
#SBATCH --gres=gpu:t4:2
#SBATCH --mem=20G
#SBATCH --mail-type=R,END,FAIL # Send email on job END and FAIL
#SBATCH --mail-user=soroush1@yorku.ca

srun python /home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/scripts/imagenet_script_vgg19.py