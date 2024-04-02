#!/bin/bash

# python inference.py --output ../prediction --batch 24 --model ../models/class9_m_synapse.pth --cfg configs/m.yml --data /root/autodl-tmp/Tools/image.nii.gz

# python inference.py --output ../predictions --batch 24 --model ../models/finetune_2.pth --cfg configs/m.yml --data /root/autodl-tmp/Dataset701_AbdomenCT/imagesVal/FLARETs_0014_0000.nii.gz --label /root/autodl-tmp/Dataset701_AbdomenCT/labelsVal/FLARETs_0014.nii.gz

python inference.py --output ../prediction_224 --batch 24 --model ../models_224/unet_2.pth --cfg configs/s.yml --data /root/autodl-tmp/Dataset701_AbdomenCT/imagesTr/FLARE22_Tr_0002_0000.nii.gz