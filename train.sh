#!/bin/bash

python train.py --cfg configs/m.yml --dataset ../Data701 --epochs 20 --output model --base_lr 0.02 --batch_size 12 --log_dir /root/tf-logs --pretrained ../model/finetune_1.pth

# python train.py --cfg configs/m.yml --dataset ../Data701 --epochs 20 --output model --base_lr 0.05 --batch_size 12 --log_dir /root/tf-logs

# python train.py --cfg configs/tiny.yml --dataset ../data_test --epochs 100 --output model --base_lr 0.0001 --batch_size 1