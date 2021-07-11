#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: zhao zhishan
# Created Time : 2021-01-16 15:56
# File Name: models/dcn.py
# Description:
"""
import tensorflow as tf
from models.base_model import BaseModel
from models.layers import dnn_layer, dropout_layer


class ModelConfig:
    cross_layer = 3
    hidden_units = [200, 200, 200]


def cross_layer(x, layer_num):
    x0 = x
    xi = x
    input_dim = x.get_shape().as_list()[-1]
    for i in range(layer_num):
        w = tf.get_variable('cross_weight_{}'.format(i), shape=[input_dim], trainable=True)
        b = tf.get_variable('cross_bias_{}'.format(i), shape=[input_dim], trainable=True)
        xw = tf.reduce_sum(xi * w, axis=1, keep_dims=True)
        xi = x0 * xw + b + xi
    return xi



class Model(BaseModel):
    def __init__(self, vocab_size, field_num, params):
        super(Model, self).__init__(vocab_size, field_num, params)
        self._model_config = ModelConfig

    def _model_fn(self, feat_idx, feat_val):
        emb_dim = self._params['emb_size']
        field_num = self._field_num
        l2_reg = self._params['l2_reg']
        with tf.variable_scope('dcn', reuse=tf.AUTO_REUSE):
            feat_idx = tf.reshape(feat_idx, [-1, field_num])
            feat_val = tf.reshape(feat_val, [-1, field_num])
            embedding_matrix = tf.get_variable('feature_embedding',
                                               shape=[self._vocab_size, emb_dim],
                                               initializer=tf.uniform_unit_scaling_initializer(),
                                               trainable=True)
            with tf.device("/cpu:0"):
                feat_emb = tf.nn.embedding_lookup(params=embedding_matrix, ids=feat_idx)
            # emb regularization
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, feat_emb)
            feat_val = tf.reshape(feat_val, [-1, field_num, 1])
            model_input = feat_emb * feat_val
            model_input = tf.reshape(model_input, shape=[-1, field_num*emb_dim])
            # cross
            cross_output = cross_layer(model_input, self._model_config.cross_layer)
            # dnn
            fc_output = dnn_layer(model_input,
                                  hidden_units=self._model_config.hidden_units,
                                  activation=tf.nn.relu,
                                  l2_reg=l2_reg,
                                  dropout_rate=self._params['dropout_rate'],
                                  is_train=self.is_train)
            # concat
            final_output = tf.concat([fc_output, cross_output], axis=1)
            final_output = tf.layers.dense(final_output, 1,
                                           use_bias=False,
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
            logits = tf.reshape(final_output, shape=[-1])
            scores = tf.sigmoid(logits)
            return logits, scores
