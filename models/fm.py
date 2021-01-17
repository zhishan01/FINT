#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: zhao zhishan
# Created Time : 2021-01-16 04:37
# File Name: fm.py
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
        emb_dim = self._params['emb_size']
        field_num = self._field_num
        l2_reg = self._params['l2_reg']
        with tf.variable_scope('fm', reuse=tf.AUTO_REUSE):
            feat_idx = tf.reshape(feat_idx, [-1, field_num])
            feat_val = tf.reshape(feat_val, [-1, field_num])
            weight_matrix = tf.get_variable('linear_weight',
                                             shape=[self._vocab_size],
                                             trainable=True)
            bias = tf.get_variable('bias',
                                   shape=[1],
                                   initializer=tf.zeros_initializer(),
                                   trainable=True)
            embedding_matrix = tf.get_variable('feature_embedding',
                                               shape=[self._vocab_size, emb_dim],
                                               initializer=tf.uniform_unit_scaling_initializer(),
                                               trainable=True)
            with tf.device("/cpu:0"):
                linear_weight = tf.nn.embedding_lookup(params=weight_matrix, ids=feat_idx)
                feat_emb = tf.nn.embedding_lookup(params=embedding_matrix, ids=feat_idx)
            # emb regularization
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, linear_weight)
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, feat_emb)
            # first order
            first_order_output = tf.reduce_sum(tf.multiply(feat_val, linear_weight), axis=1)
            # second order
            feat_val = tf.reshape(feat_val, [-1, field_num, 1])
            model_input = tf.multiply(feat_emb, feat_val)
            square_sum = tf.square(tf.reduce_sum(model_input, axis=1))
            sum_square = tf.reduce_sum(tf.square(model_input), axis=1)
            second_order_output = tf.reduce_sum(0.5*(tf.subtract(square_sum, sum_square)), axis=1)
            # sum
            logits = tf.add_n([first_order_output, second_order_output]) + bias
            scores = tf.sigmoid(logits)
            return logits, scores
