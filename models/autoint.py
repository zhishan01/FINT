#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: zhao zhishan
# Created Time : 2021-01-17 03:53
# File Name: models/autoint.py
# Description:
"""
import tensorflow as tf
from models.base_model import BaseModel
from models.layers import dnn_layer, dropout_layer, deep_layer

class ModelConfig:
    deep_layers = [200,200,200]
    blocks = 3
    block_shape = [32, 32, 32]
    heads = 2
    has_residual = True


def normalize(inputs, epsilon=1e-8):
    '''
    Applies layer normalization
    Args:
        inputs: A tensor with 2 or more dimensions
        epsilon: A floating number to prevent Zero Division
    Returns:
        A tensor with the same shape and data dtype
    '''
    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:]

    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    beta = tf.Variable(tf.zeros(params_shape))
    gamma = tf.Variable(tf.ones(params_shape))
    normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
    outputs = gamma * normalized + beta

    return outputs


def multihead_attention(queries,
                        keys,
                        values,
                        num_units=None,
                        num_heads=1,
                        dropout_rate=0,
                        is_training=True,
                        has_residual=True):
    '''
    part of the code comes from: https://github.com/DeepGraphLearning/RecommenderSystems/tree/master/featureRec
    '''

    if num_units is None:
        num_units = queries.get_shape().as_list[-1]

    # Linear projections
    Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)
    K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)
    V = tf.layers.dense(values, num_units, activation=tf.nn.relu)
    if has_residual:
        V_res = tf.layers.dense(values, num_units, activation=tf.nn.relu)

    # Split and concat
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

    # Multiplication
    weights = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))

    # Scale
    weights = weights / (K_.get_shape().as_list()[-1] ** 0.5)

    # Activation
    weights = tf.nn.softmax(weights)

    # Dropouts
    weights = tf.layers.dropout(weights, rate=dropout_rate,
                                training=tf.convert_to_tensor(is_training))

    # Weighted sum
    outputs = tf.matmul(weights, V_)

    # Restore shape
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)

    # Residual connection
    if has_residual:
        outputs += V_res
    outputs = tf.nn.relu(outputs)
    # Normalize
    outputs = normalize(outputs)
    return outputs


class Model(BaseModel):
    def __init__(self, vocab_size, field_num, params):
        super(Model, self).__init__(vocab_size, field_num, params)
        self._model_config = ModelConfig

    def _model_fn(self, feat_idx, feat_val):
        emb_dim = self._params['emb_size']
        field_num = self._field_num
        l2_reg = self._params['l2_reg']
        with tf.variable_scope('autoint', reuse=tf.AUTO_REUSE):
            feat_idx = tf.reshape(feat_idx, [-1, field_num])
            feat_val = tf.reshape(feat_val, [-1, field_num])
            embedding_matrix = tf.get_variable('feature_embedding',
                                               shape=[self._vocab_size, emb_dim],
                                               initializer=tf.uniform_unit_scaling_initializer(),
                                               trainable=True)
            with tf.device("/cpu:0"):
                feat_emb = tf.nn.embedding_lookup(params=embedding_matrix, ids=feat_idx)
            feat_emb = dropout_layer(feat_emb, self._params['dropout_rate'], self.is_train)
            # emb regularization
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, feat_emb)
            feat_val = tf.reshape(feat_val, [-1, field_num, 1])
            model_input = feat_emb * feat_val
            # joint training with feedforward nn
            deep_logits = None
            if self._model_config.deep_layers != None:
                fc_input = tf.reshape(model_input, shape=[-1, field_num*emb_dim])
                deep_logits = deep_layer(fc_input,
                                         hidden_units=self._model_config.deep_layers,
                                         activation=tf.nn.relu,
                                         l2_reg=l2_reg,
                                         dropout_rate=self._params['dropout_rate'],
                                         is_train=self.is_train,
                                         output_bias=True)
            # ---------- main part of AutoInt-------------------
            block_shape = self._model_config.block_shape
            xi = model_input
            for i in range(self._model_config.blocks):
                xi = multihead_attention(queries=xi,
                                         keys=xi,
                                         values=xi,
                                         num_units=block_shape[i],
                                         num_heads=self._model_config.heads,
                                         dropout_rate=self._params['dropout_rate'],
                                         is_training=self.is_train,
                                         has_residual=self._model_config.has_residual)
            flat = tf.reshape(xi, shape=[-1, block_shape[-1] * field_num])
            logits = tf.layers.dense(flat, 1,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
            logits = tf.reshape(logits, shape=[-1])
            if self._model_config.deep_layers != None:
                logits += deep_logits
            scores = tf.sigmoid(logits)
        return logits, scores
