#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: zhao zhishan
# Created Time : 2021-01-17 05:21
# File Name: models/xdeepfm.py
# Description:
"""
import tensorflow as tf
from models.base_model import BaseModel
from models.layers import deep_layer, dropout_layer, active_layer

class ModelConfig:
    cross_layer_sizes = [200, 200, 200]
    cross_activation = 'identity'
    hidden_units = [200, 200, 200]


def cin_layer(nn_input, field_num, emb_dim, dropout_rate, is_train, l2_reg, hparams,
              res=False, direct=False, bias=False, reduce_D=False, f_dim=2):
    '''
    part of the code comes from: https://github.com/Leavingseason/xDeepFM
    '''
    hidden_nn_layers = []
    field_nums = []
    final_len = 0
    field_nums.append(int(field_num))
    hidden_nn_layers.append(nn_input)
    final_result = []
    split_tensor0 = tf.split(hidden_nn_layers[0], emb_dim * [1], 2)
    with tf.variable_scope("exfm_part") as scope:
        for idx, layer_size in enumerate(hparams.cross_layer_sizes):
            split_tensor = tf.split(hidden_nn_layers[-1], emb_dim * [1], 2)
            dot_result_m = tf.matmul(split_tensor0, split_tensor, transpose_b=True)
            dot_result_o = tf.reshape(dot_result_m, shape=[emb_dim, -1, field_nums[0]*field_nums[-1]])
            dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])

            if reduce_D:
                filters0 = tf.get_variable("f0_" + str(idx),
                                           shape=[1, layer_size, field_nums[0], f_dim],
                                           regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                           dtype=tf.float32)
                filters_ = tf.get_variable("f__" + str(idx),
                                           shape=[1, layer_size, f_dim, field_nums[-1]],
                                           regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                           dtype=tf.float32)
                filters_m = tf.matmul(filters0, filters_)
                filters_o = tf.reshape(filters_m, shape=[1, layer_size, field_nums[0] * field_nums[-1]])
                filters = tf.transpose(filters_o, perm=[0, 2, 1])
            else:
                filters = tf.get_variable(name="f_"+str(idx),
                                     shape=[1, field_nums[-1]*field_nums[0], layer_size],
                                     regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                     dtype=tf.float32)
            curr_out = tf.nn.conv1d(dot_result, filters=filters, stride=1, padding='VALID')

            # BIAS ADD
            if bias:
                b = tf.get_variable(name="f_b" + str(idx),
                                shape=[layer_size],
                                dtype=tf.float32,
                                regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                initializer=tf.zeros_initializer())
                curr_out = tf.nn.bias_add(curr_out, b)

            curr_out = active_layer(curr_out, hparams.cross_activation)

            curr_out = tf.transpose(curr_out, perm=[0, 2, 1])

            if direct:
                direct_connect = curr_out
                next_hidden = curr_out
                final_len += layer_size
                field_nums.append(int(layer_size))

            else:
                if idx != len(hparams.cross_layer_sizes) - 1:
                    next_hidden, direct_connect = tf.split(curr_out, 2 * [int(layer_size / 2)], 1)
                    final_len += int(layer_size / 2)
                else:
                    direct_connect = curr_out
                    next_hidden = 0
                    final_len += layer_size
                field_nums.append(int(layer_size / 2))

            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)

        result = tf.concat(final_result, axis=1)
        result = tf.reduce_sum(result, -1)
        if res:
            w_nn_output1 = tf.get_variable(name='w_nn_output1',
                                           shape=[final_len, 128],
                                           regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                           dtype=tf.float32)
            b_nn_output1 = tf.get_variable(name='b_nn_output1',
                                           shape=[128],
                                           regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                           dtype=tf.float32,
                                           initializer=tf.zeros_initializer())
            exFM_out0 = tf.nn.xw_plus_b(result, w_nn_output1, b_nn_output1)
            exFM_out0 = dropout_layer(exFM_out0, dropout_rate, is_train)
            exFM_out1 = active_layer(logit=exFM_out0, activation="relu")
            w_nn_output2 = tf.get_variable(name='w_nn_output2',
                                           shape=[128 + final_len, 1],
                                           regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                           dtype=tf.float32)
            b_nn_output2 = tf.get_variable(name='b_nn_output2',
                                           shape=[1],
                                           regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                           dtype=tf.float32,
                                           initializer=tf.zeros_initializer())
            exFM_in = tf.concat([exFM_out1, result], axis=1, name="user_emb")
            exFM_out = tf.nn.xw_plus_b(exFM_in, w_nn_output2, b_nn_output2)

        else:
            w_nn_output = tf.get_variable(name='w_nn_output',
                                          shape=[final_len, 1],
                                          regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                          dtype=tf.float32)
            b_nn_output = tf.get_variable(name='b_nn_output',
                                          shape=[1],
                                          regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                          dtype=tf.float32,
                                          initializer=tf.zeros_initializer())
            exFM_out = tf.nn.xw_plus_b(result, w_nn_output, b_nn_output)
        exFM_out = tf.reshape(exFM_out, shape=[-1])
        return exFM_out


class Model(BaseModel):
    def __init__(self, vocab_size, field_num, params):
        super(Model, self).__init__(vocab_size, field_num, params)
        self._model_config = ModelConfig

    def _model_fn(self, feat_idx, feat_val):
        emb_dim = self._params['emb_size']
        field_num = self._field_num
        l2_reg = self._params['l2_reg']
        dropout_rate = self._params['dropout_rate']
        with tf.variable_scope('xdeepfm', reuse=tf.AUTO_REUSE):
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
            # linear
            linear_logits = tf.reduce_sum(tf.multiply(feat_val, linear_weight), axis=1) + bias
            # dnn
            feat_val = tf.reshape(feat_val, [-1, field_num, 1])
            model_input = tf.multiply(feat_emb, feat_val)
            fc_input = tf.reshape(model_input, shape=[-1, field_num*emb_dim])
            deep_logits = deep_layer(fc_input,
                                     hidden_units=self._model_config.hidden_units,
                                     activation=tf.nn.relu,
                                     l2_reg=l2_reg,
                                     dropout_rate=self._params['dropout_rate'],
                                     is_train=self.is_train,
                                     output_bias=True)
            # cin
            cin_logits = cin_layer(model_input, field_num, emb_dim,
                                   dropout_rate, self.is_train, l2_reg, self._model_config,
                                   res=False, direct=False, bias=False, reduce_D=False, f_dim=2)
            # sum
            logits = tf.add_n([linear_logits, deep_logits, cin_logits])
            scores = tf.sigmoid(logits)
            return logits, scores
