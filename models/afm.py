#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: zhao zhishan
# Created Time : 2021-01-16 07:07
# File Name: models/afm.py
# Description:
"""
import tensorflow as tf
from models.base_model import BaseModel
from models.layers import dnn_layer, dropout_layer


class ModelConfig:
    hidden_factor = [200, 200, 200]


def afm_attention_layer(x, field_num, hidden_factor, dropout_rate, l2_reg, is_train):
    '''
    part of the code comes from: https://github.com/hexiangnan/attentional_factorization_machine
    '''
    element_wise_product_list = []
    for i in range(0, field_num):
        for j in range(i+1, field_num):
            element_wise_product_list.append(tf.multiply(x[:,i,:], x[:,j,:]))
    element_wise_product = tf.stack(element_wise_product_list) # (M'*(M'-1)) * None * K
    element_wise_product = tf.transpose(element_wise_product, perm=[1,0,2], name="element_wise_product") # None * (M'*(M'-1)) * K
    # _________ MLP Layer / attention part _____________
    attention_mul = tf.layers.dense(element_wise_product,
                                    hidden_factor[1],
                                    activation=tf.nn.relu,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
    attention_p = tf.get_variable('attention_p', shape=[hidden_factor[0]], trainable=True)
    attention_relu = tf.reduce_sum(tf.multiply(attention_p, attention_mul), 2, keep_dims=True) # None * (M'*(M'-1)) * 1
    attention_val = tf.nn.softmax(attention_relu)
    attention_val = dropout_layer(attention_val, dropout_rate, is_train)
    # _________ Attention-aware Pairwise Interaction Layer _____________
    AFM = tf.reduce_sum(tf.multiply(attention_val, element_wise_product), 1, name="afm") # None * K
    return AFM



class Model(BaseModel):
    def __init__(self, vocab_size, field_num, params):
        super(Model, self).__init__(vocab_size, field_num, params)
        self._model_config = ModelConfig

    def _model_fn(self, feat_idx, feat_val):
        emb_dim = self._params['emb_size']
        field_num = self._field_num
        l2_reg = self._params['l2_reg']
        dropout_rate = self._params['dropout_rate']
        with tf.variable_scope('afm', reuse=tf.AUTO_REUSE):
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
            first_order_output = tf.reduce_sum(feat_val * linear_weight, axis=1)
            # second order
            feat_val = tf.reshape(feat_val, [-1, field_num, 1])
            model_input = feat_emb * feat_val
            attention_output = afm_attention_layer(
                model_input, field_num, self._model_config.hidden_factor, dropout_rate, l2_reg, self.is_train)
            attention_output = dropout_layer(attention_output, dropout_rate, self.is_train)
            # dnn
            fc_output = tf.layers.dense(attention_output, 1,
                                        use_bias=False,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
            fc_output = tf.reshape(fc_output, shape=[-1])
            # sum
            logits = first_order_output + fc_output + bias
            scores = tf.sigmoid(logits)
            return logits, scores
