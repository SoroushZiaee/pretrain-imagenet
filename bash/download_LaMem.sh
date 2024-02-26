#!/bin/bash

cd /home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/data/LaMem
wget -v http://memorability.csail.mit.edu/download_lamem.tar.gz

mkdir /home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/data/LaMem/lamem_images

pv /home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/data/LaMem/download_lamem.tar.gz | tar -xzf - -C /home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/data/LaMem/lamem_images/
