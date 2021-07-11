#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: zhao zhishan
# Created Time : 2021-01-14 10:07
# File Name: models/fint.py
# Description:
"""
import tensorflow as tf
from models.base_model import BaseModel
from models.layers import deep_layer


class ModelConfig:
    fint_layer = 1
    hidden_units = [200, 200, 200]


def field_aware_interaction_layer(xi, x0, field_num, l2_reg):
    x = tf.pad(x0, paddings=[[0,0], [0,1], [0,0]], constant_values=1)
    x = tf.transpose(x, perm=[0,2,1])
    context_vec = tf.layers.dense(x, field_num, use_bias=False,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
    context_vec = tf.transpose(context_vec, perm=[0,2,1])
    return xi * context_vec


def fint_interaction_layer(x, field_num, layers,  l2_reg):
    x0 = x
    xi = x
    for i in range(layers):
        xi = field_aware_interaction_layer(xi, x0, field_num, l2_reg)
    return xi


class Model(BaseModel):
    def __init__(self, vocab_size, field_num, params):
        super(Model, self).__init__(vocab_size, field_num, params)
        self._model_config = ModelConfig

    def _model_fn(self, feat_idx, feat_val):
        emb_dim = self._params['emb_size']
        field_num = self._field_num
        l2_reg = self._params['l2_reg']
        with tf.variable_scope('fint', reuse=tf.AUTO_REUSE):
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
            fint_output = fint_interaction_layer(
                model_input, field_num, self._model_config.fint_layer, l2_reg)
            fc_input = tf.reshape(fint_output, shape=[-1, field_num*emb_dim])
            logits = deep_layer(fc_input,
                                hidden_units=self._model_config.hidden_units,
                                activation=tf.nn.relu,
                                l2_reg=l2_reg,
                                dropout_rate=self._params['dropout_rate'],
                                is_train=self.is_train,
                                output_bias=True)
            scores = tf.sigmoid(logits)
        return logits, scores
