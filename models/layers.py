#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: zhao zhishan
# Created Time : 2021-01-14 11:11
# File Name: models/layers.py
# Description:
"""
import tensorflow as tf

def batch_normalize(x, is_train, axis=-1):
    return tf.layers.batch_normalization(x, axis=axis, training=is_train)

def layer_normalize(inputs, epsilon=1e-8):
    '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
    inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:]

    mean, variance = tf.nn.moments(inputs, [-2, -1], keep_dims=True)
    tf.logging.info('mean shape:{}'.format(mean.get_shape().as_list()))
    beta= tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
    gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
    normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
    #normalized = inputs
    outputs = gamma * normalized + beta

    return outputs

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
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                    bias_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
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
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                  bias_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
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
