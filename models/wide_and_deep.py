#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: zhao zhishan
# Created Time : 2021-01-16 10:21
# File Name: models/wide_and_deep.py
# Description:
"""
import tensorflow as tf
from models.base_model import BaseModel
from models.layers import deep_layer

class ModelConfig:
    hidden_units = [200, 200, 200]

class Model(BaseModel):
    def __init__(self, vocab_size, field_num, params):
        super(Model, self).__init__(vocab_size, field_num, params)
        self._model_config = ModelConfig

    def _model_fn(self, feat_idx, feat_val):
        emb_dim = self._params['emb_size']
        field_num = self._field_num
        l2_reg = self._params['l2_reg']
        with tf.variable_scope('wide_and_deep', reuse=tf.AUTO_REUSE):
            feat_idx = tf.reshape(feat_idx, [-1, field_num])
            feat_val = tf.reshape(feat_val, [-1, field_num])
            weight_matrix = tf.get_variable('wide_weight',
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
                wide_weight = tf.nn.embedding_lookup(params=weight_matrix, ids=feat_idx)
                feat_emb = tf.nn.embedding_lookup(params=embedding_matrix, ids=feat_idx)
            # emb regularization
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, wide_weight)
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, feat_emb)
            # wide part
            wide_logits = tf.reduce_sum(tf.multiply(feat_val, wide_weight), axis=1)
            # deep part
            feat_val = tf.reshape(feat_val, [-1, field_num, 1])
            model_input = tf.multiply(feat_emb, feat_val)
            fc_input = tf.reshape(model_input, shape=[-1, field_num*emb_dim])
            deep_logits = deep_layer(fc_input,
                                     hidden_units=self._model_config.hidden_units,
                                     activation=tf.nn.relu,
                                     l2_reg=l2_reg,
                                     dropout_rate=self._params['dropout_rate'],
                                     is_train=self.is_train,
                                     output_bias=False)
            logits = tf.add_n([wide_logits, deep_logits]) + bias
            scores = tf.sigmoid(logits)
        return logits, scores
