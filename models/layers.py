#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: zhao zhishan
# Created Time : 2021-01-14 11:11
# File Name: models/layers.py
# Description:
"""
import tensorflow as tf


def dropout_layer(x, dropout_rate, is_train):
    output = x
    if dropout_rate > 0:
        output = tf.layers.dropout(x,
                                   rate=dropout_rate,
                                   training=is_train)
    return output


def dnn_layer(x, hidden_units, activation, l2_reg, dropout_rate, is_train):
    fc_input = x
    fc_output = None
    for unit in hidden_units:
        fc_output = tf.layers.dense(fc_input,
                                    unit,
                                    activation=activation,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
        fc_output = dropout_layer(fc_output, dropout_rate, is_train)
        fc_input = fc_output
    return fc_output


def deep_layer(fc_input, hidden_units, activation, l2_reg, dropout_rate, is_train, output_bias):
    fc_output = dnn_layer(fc_input,
                          hidden_units=hidden_units,
                          activation=tf.nn.relu,
                          l2_reg=l2_reg,
                          dropout_rate=dropout_rate,
                          is_train=is_train)
    deep_logits = tf.layers.dense(fc_output, 1,
                                  use_bias=output_bias,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
    deep_logits = tf.reshape(deep_logits, shape=[-1])
    return deep_logits

def active_layer(x, activation):
    if activation == 'sigmoid':
        return tf.nn.sigmoid(x)
    elif activation == 'softmax':
        return tf.nn.softmax(x)
    elif activation == 'relu':
        return tf.nn.relu(x)
    elif activation == 'tanh':
        return tf.nn.tanh(x)
    elif activation == 'elu':
        return tf.nn.elu(x)
    elif activation == 'identity':
        return tf.identity(x)
    else:
        raise ValueError("this activations not defined {0}".format(activation))
