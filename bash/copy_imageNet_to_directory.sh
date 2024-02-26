#!/bin/bash

cp -r /datashare/ImageNet/ILSVRC2012/ILSVRC2012_devkit_t12 /home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/data/ImageNet/ImageNet/
cp /datashare/ImageNet/ILSVRC2012/ILSVRC2012_img_train.tar /home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/data/ImageNet/ImageNet/
cp /datashare/ImageNet/ILSVRC2012/ILSVRC2012_img_val.tar /home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/data/ImageNet/ImageNet/
mkdir /home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/data/ImageNet/ImageNet/ILSVRC2012_img_train
mkdir /home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/data/ImageNet/ImageNet/ILSVRC2012_img_val
pv /home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/data/ImageNet/ImageNet/ILSVRC2012_img_val.tar | tar -xf - -C /home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/data/ImageNet/ImageNet/ILSVRC2012_img_val
pv /home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/data/ImageNet/ImageNet/ILSVRC2012_img_train.tar | tar -xf - -C /home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/data/ImageNet/ImageNet/ILSVRC2012_img_train