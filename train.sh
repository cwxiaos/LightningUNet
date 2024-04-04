#!/bin/bash

# CPU Intel Xeon Gold 6430, RAM 240GB, GPU RTX4090 x 2, Ubuntu 20.04, CUDA 11.3, torch 1.11

# 这组参数导致过拟合, Those parameters lead to overfit
# batch取太大会陷入局部最小值, batch取太小会抖动厉害?
# python train.py --cfg configs/s.yml --dataset ../data --epochs 60 --output ../model --base_lr 0.2 --batch_size 48 --log_dir /root/tf-logs --n_gpu 2

# CPU Intel Xeon Gold 6430, RAM 120GB, GPU RTX4090 x 1, Ubuntu 20.04, CUDA 11.3, torch 1.11
python train.py --cfg configs/s.yml --dataset ../data --epochs 45 --output ../model --base_lr 0.05 --batch_size 48 --log_dir /root/tf-logs
# python train.py --cfg configs/s.yml --dataset ../Dataset --epochs 15 --output ../model --base_lr 0.03 --batch_size 12 --log_dir /root/tf-logs --pretrained ../models/best_1.pth
# python train.py --cfg configs/s.yml --dataset ../Dataset --epochs 15 --output ../model --base_lr 0.01 --batch_size 6 --log_dir /root/tf-logs --pretrained ../models/best_2.pth

# RTX4090 x 1, RAM 120GB, Ubuntu20.04, CUDA 11.3, torch 1.11
# python train.py --cfg configs/m.yml --dataset ../Data701 --epochs 20 --output model --base_lr 0.05 --batch_size 12 --log_dir /root/tf-logs

# RTX4090 x 1, RAM 120GB, Ubuntu20.04, CUDA 11.3, torch 1.11
# python train.py --cfg configs/tiny.yml --dataset ../data_test --epochs 100 --output model --base_lr 0.0001 --batch_size 1