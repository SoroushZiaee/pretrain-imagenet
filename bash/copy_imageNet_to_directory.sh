#!/bin/bash

cp -r /datashare/ImageNet/ILSVRC2012/ILSVRC2012_devkit_t12 /home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/data/ImageNet/ImageNet/
cp /datashare/ImageNet/ILSVRC2012/ILSVRC2012_img_train.tar /home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/data/ImageNet/ImageNet/
cp /datashare/ImageNet/ILSVRC2012/ILSVRC2012_img_val.tar /home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/data/ImageNet/ImageNet/

cd /home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/data/ImageNet/ImageNet/
tar -czvf /home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/data/ImageNet/ImageNet/ILSVRC2012_devkit_t12.tar.gz /home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/data/ImageNet/ImageNet/ILSVRC2012_devkit_t12
