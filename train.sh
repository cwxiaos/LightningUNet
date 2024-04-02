#!/bin/bash

python train.py --cfg configs/m.yml --dataset ../Data701 --epochs 50 --output ../model_512 --base_lr 0.1 --batch_size 12 --log_dir ../logs_512

# python train.py --cfg configs/m.yml --dataset ../Data701 --epochs 20 --output model --base_lr 0.05 --batch_size 12 --log_dir /root/tf-logs

# python train.py --cfg configs/tiny.yml --dataset ../data_test --epochs 100 --output model --base_lr 0.0001 --batch_size 1