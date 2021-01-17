#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: zhao zhishan
# Created Time : 2021-01-17 08:09
# File Name: models/lr.py
# Description:
"""
import tensorflow as tf
from models.base_model import BaseModel

class ModelConfig:
    pass

class Model(BaseModel):
    def __init__(self, vocab_size, field_num, params):
        super(Model, self).__init__(vocab_size, field_num, params)
        self._model_config = ModelConfig

    def _model_fn(self, feat_idx, feat_val):
        field_num = self._field_num
        l2_reg = self._params['l2_reg']
        with tf.variable_scope('logistic_regression', reuse=tf.AUTO_REUSE):
            feat_idx = tf.reshape(feat_idx, [-1, field_num])
            feat_val = tf.reshape(feat_val, [-1, field_num])
            weight_matrix = tf.get_variable('linear_weight',
                                             shape=[self._vocab_size],
                                             trainable=True)
            bias = tf.get_variable('bias',
                                   shape=[1],
                                   initializer=tf.zeros_initializer(),
                                   trainable=True)
            with tf.device("/cpu:0"):
                linear_weight = tf.nn.embedding_lookup(params=weight_matrix, ids=feat_idx)
            # regularization
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, linear_weight)
            logits = tf.reduce_sum(tf.multiply(feat_val, linear_weight), axis=1) + bias
            scores = tf.sigmoid(logits)
        return logits, scores
