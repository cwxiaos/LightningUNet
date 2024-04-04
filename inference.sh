#!/bin/bash

# python inference.py --output ../prediction --batch 24 --model ../models/class9_m_synapse.pth --cfg configs/m.yml --data /root/autodl-tmp/Tools/image.nii.gz

# python inference.py --output ../predictions --batch 24 --model ../models/finetune_2.pth --cfg configs/m.yml --data /root/autodl-tmp/Dataset701_AbdomenCT/imagesVal/FLARETs_0014_0000.nii.gz --label /root/autodl-tmp/Dataset701_AbdomenCT/labelsVal/FLARETs_0014.nii.gz

python inference.py --cfg configs/s.yml --output ../prediction --batch 3 --model /root/autodl-tmp/model/test.pth --data /root/autodl-tmp/val/image.nii.gz

# python inference.py --output ../prediction --batch 48 --model /root/autodl-tmp/model/best.pth --cfg configs/s.yml --data ../val/image.nii.gz