#!/bin/bash

# Before Out of GRAM, A Larger Batch Can Make Inference More Quick.
python inference.py --cfg configs/s.yml --output ../prediction --batch 3 --model ../model/test.pth --label ../val/label.nii.gz --data ../val/image.nii.gz