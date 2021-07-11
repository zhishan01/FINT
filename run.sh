#!/bin/sh
# Author: zhao zhishan
# Created Time : 2021-01-14 11:26
# File Name: run.sh
# Description:

model_name=fint
#data=avazu_tmp
data=criteo
#data=kdd2012
version=fint3
gpu=3
field_num=39
#field_num=21
#field_num=13

python train.py --feature_size_file ./data/${data}/data.feat_info \
    --train_data ./data/${data}/data.train \
    --val_data ./data/${data}/data.val \
    --eval_data ./data/${data}/data.eval \
    --test_data ./data/${data}/data.test \
    --model_name $model_name \
    --model_dir ./checkpoints/${version}/${data}/ \
    --learning_rate 0.001 \
    --epoch 1 \
    --l2_reg 0.0 \
    --dropout_rate 0 \
    --batch_size 1024 \
    --emb_size 16 \
    --eval_step 5000 \
    --field_num ${field_num} \
    --version ${version} \
    --gpu ${gpu} \
