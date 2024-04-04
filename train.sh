#!/bin/bash

# CPU Intel Xeon Gold 6430, RAM 120GB, GPU RTX4090 x 1, Ubuntu 20.04, CUDA 11.3, torch 1.11
python train.py --cfg configs/s.yml --dataset ../data --epochs 50 --output ../model --base_lr 0.05 --batch_size 24 --log_dir /root/tf-logs