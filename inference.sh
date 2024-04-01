#!/bin/bash

# python inference.py --output ./prediction --batch 24 --model ./models/finetune_1.pth --cfg configs/m.yml --data /root/autodl-tmp/Dataset701_AbdomenCT/imagesVal/FLARETs_0009_0000.nii.gz

# python inference.py --output ../predictions --batch 24 --model ../models/finetune_2.pth --cfg configs/m.yml --data /root/autodl-tmp/Dataset701_AbdomenCT/imagesVal/FLARETs_0014_0000.nii.gz --label /root/autodl-tmp/Dataset701_AbdomenCT/labelsVal/FLARETs_0014.nii.gz

python inference.py --output ./prediction --batch 24 --model ../models/class14_m_701.pth --cfg configs/m.yml --data /root/autodl-tmp/Dataset701_AbdomenCT/imagesVal/FLARETs_0014_0000.nii.gz