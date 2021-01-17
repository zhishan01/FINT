#!/bin/sh
# Author: zhao zhishan
# Created Time : 2021-01-14 11:26
# File Name: run.sh
# Description:
python train.py --feature_size_file './data/criteo/data.feat_info' \
    --train_data './data/criteo/data.train' \
    --val_data './data/criteo/data.val' \
    --eval_data './data/criteo/data.eval' \
    --test_data './data/criteo/data.test' \
    --model_name fint \
    --model_dir './checkpoints/fint/criteo' \
    --learning_rate 0.001 \
    --epoch 30 \
    --l2_reg 0.0 \
    --dropout_rate 0 \
    --batch_size 1024 \
    --emb_size 16 \
    --eval_step 10000 \
    --field_num 39 \
