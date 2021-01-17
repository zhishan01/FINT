#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: zhao zhishan
# Created Time : 2021-01-15 03:31
# File Name: data_loader.py
# Description:
"""
import tensorflow as tf

def parse_example(example):
    feature_description = {
        'label': tf.FixedLenFeature([], tf.int64),
        'feat_idx': tf.VarLenFeature(tf.int64),
        'feat_val': tf.VarLenFeature(tf.float32),
    }
    feature_dict = tf.io.parse_example(example, feature_description)
    label = tf.cast(feature_dict['label'], dtype=tf.float32)
    norm_val = tf.square(tf.log(feature_dict['feat_val'].values + tf.constant(1e-6, dtype=tf.float32)))
    norm_val2 = tf.maximum(tf.constant(2, dtype=tf.float32), norm_val)
    feat_val = tf.minimum(feature_dict['feat_val'].values, norm_val2)
    return label, feature_dict['feat_idx'].values, feat_val


def build_dataset(file_name, epoch, batch_size):
    dataset = tf.data.TFRecordDataset(file_name, num_parallel_reads=8)
    dataset = dataset.repeat(epoch).batch(batch_size)
    dataset = dataset.map(lambda x: parse_example(x), num_parallel_calls=8)
    dataset = dataset.prefetch(buffer_size=1)
    return dataset


def create_dataset_iterator(train_dataset, val_dataset, eval_dataset, test_dataset):
    train_data_iter = train_dataset.make_one_shot_iterator()
    val_data_iter= val_dataset.make_initializable_iterator()
    eval_data_iter= eval_dataset.make_initializable_iterator()
    test_data_iter = test_dataset.make_initializable_iterator()

    handle = tf.placeholder(tf.string,shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types)
    batch_data = iterator.get_next()
    return handle, batch_data, train_data_iter, val_data_iter, eval_data_iter, test_data_iter


def get_vocab_size(file_name):
    with open(file_name) as rf:
        vocab_size = int(rf.readline().strip())
    return vocab_size
