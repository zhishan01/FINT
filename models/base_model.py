#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: zhao zhishan
# Created Time : 2021-01-15 07:20
# File Name: models/base_model.py
# Description:
"""
import tensorflow as tf

class BaseModel(object):
    def __init__(self, vocab_size, field_num, params):
        self._vocab_size = vocab_size
        self._field_num = field_num
        self._params = params

    def init_graph(self, y_true, feat_idx, feat_val, handle):
        self.y_true = y_true
        self.feat_idx = feat_idx
        self.feat_val = feat_val
        self.handle = handle
        self.is_train = tf.placeholder(tf.bool, name="is_train")
        self.logits, self.scores = self._model_fn(feat_idx, feat_val)
        self.loss = self._compute_loss(y_true, self.logits)
        self.train_op = self._build_train_op(self.loss)
        self.saver = tf.train.Saver(max_to_keep=1)

    def _model_fn(self, feat_idx, feat_val):
        raise NotImplementedError('model must implement _model_fn methold')

    def _compute_loss(self, labels, logits):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
        if self._params['l2_reg'] > 0:
            # include emb l2 reg loss
            regularizer = tf.contrib.layers.l2_regularizer(self._params['l2_reg'])
            tf.contrib.layers.apply_regularization(regularizer)
            l2_reg_loss = tf.losses.get_regularization_loss()
            loss += l2_reg_loss
        return loss

    def _build_train_op(self, loss):
        global_step = tf.train.get_or_create_global_step()
        self.lr = tf.train.exponential_decay(learning_rate = self._params['lr'],
                                             global_step=global_step,
                                             decay_steps = 3000,
                                             decay_rate = 0.9)
        #self.lr = tf.cast(self._params['lr'], tf.float32)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            #optimizer = tf.train.AdamOptimizer(learning_rate=self._params['lr'])
            #optimizer = tf.train.AdagradOptimizer(learning_rate=self._params['lr'])
            train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op
