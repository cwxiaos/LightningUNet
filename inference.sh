#!/bin/bash

python inference.py --output ./prediction --batch 24 --model ../models/finetune_1.pth --cfg configs/m.yml --data /root/autodl-tmp/Dataset701_AbdomenCT/imagesVal/FLARETs_0009_0000.nii.gz