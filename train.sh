#!/bin/bash

# CPU Intel Xeon Gold 6430, RAM 240GB, GPU RTX4090 x 2, Ubuntu 20.04, CUDA 11.3, torch 1.11
python train.py --cfg configs/s.yml --dataset ../data --epochs 1 --output ../model --base_lr 0.2 --batch_size 48 --log_dir /root/tf-logs --n_gpu 1

# RTX4090 x 1, RAM 120GB, Ubuntu20.04, CUDA 11.3, torch 1.11
# python train.py --cfg configs/m.yml --dataset ../Data701 --epochs 20 --output model --base_lr 0.05 --batch_size 12 --log_dir /root/tf-logs

# RTX4090 x 1, RAM 120GB, Ubuntu20.04, CUDA 11.3, torch 1.11
# python train.py --cfg configs/tiny.yml --dataset ../data_test --epochs 100 --output model --base_lr 0.0001 --batch_size 1