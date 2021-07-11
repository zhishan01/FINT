#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: zhao zhishan
# Created Time : 2021-01-13 07:05
# File Name: train.py
# Description:
"""
import os
import sys
import shutil
import logging
import argparse
from importlib import import_module
import tensorflow as tf
from tensorflow.python.client import timeline
from sklearn.metrics import roc_auc_score, log_loss
from data_loader import build_dataset, create_dataset_iterator, get_vocab_size


def evaluate(data_iter, model, sess):
    data_iter_handle = sess.run(data_iter.string_handle())
    sess.run(data_iter.initializer)
    scores = []
    labels = []
    while True:
        try:
            batch_labels, batch_scores = sess.run(
                [model.y_true, model.scores],
                feed_dict={model.handle:data_iter_handle,
                           model.is_train: False})
            scores.extend(batch_scores)
            labels.extend(batch_labels)
        except tf.errors.OutOfRangeError:
            break
    auc = roc_auc_score(labels, scores)
    logloss = log_loss(labels, scores, eps=1e-7)
    return auc, logloss


def train_and_eval(train_data_iter, val_data_iter, eval_data_iter,
                   model, sess, eval_step, model_dir, print_each=1000):
    step = 0
    best_auc = 0
    last_logloss = 0
    hold_step = 0
    best_logloss = sys.maxsize
    train_handle = sess.run(train_data_iter.string_handle())
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    while True:
        if step > 0 and step % eval_step == 0:
            logging.info('=============start evaluate================')
            val_auc, val_logloss = evaluate(val_data_iter, model, sess)
            eval_auc, eval_logloss = evaluate(eval_data_iter, model, sess)
            if eval_logloss < best_logloss:
            #if eval_auc > best_auc:
                best_logloss = eval_logloss
                best_auc = eval_auc
                hold_step = 0
                model.saver.save(sess, '{}/model.ckpt'.format(model_dir), global_step=step)
            logging.info('step: {}, train auc: {:.4}, train logloss: {:.4}, eval auc: {:.4}, eval logloss: {:.4}, best_auc: {:.4}, best_logloss: {:.4}'\
                         .format(step, val_auc, val_logloss,
                                 eval_auc, eval_logloss, best_auc, best_logloss))
            logging.info('=============end evaluate================')
            if hold_step > 6:
                logging.info('eval logloss not decrease, early stopping')
                break
            hold_step += 1
        try:
            loss, _, lr = sess.run(
                [model.loss, model.train_op, model.lr],
                feed_dict={model.handle:train_handle,
                           model.is_train: True})
            if step > 0 and step % print_each == 0:
                logging.info('step: {}, train logloss: {:.4}, lr: {:.4}'.format(step, loss, lr))
        except tf.errors.OutOfRangeError:
            break
        step += 1


def run(args):
    vocab_size = get_vocab_size(args.feature_size_file)
    train_dataset = build_dataset(args.train_data, args.epoch, args.batch_size)
    val_dataset = build_dataset(args.val_data, 1, args.batch_size*5)
    eval_dataset = build_dataset(args.eval_data, 1, args.batch_size*5)
    test_dataset = build_dataset(args.test_data, 1, args.batch_size*5)
    handle, batch_data, train_data_iter, val_data_iter, eval_data_iter, test_data_iter =\
        create_dataset_iterator(train_dataset, val_dataset, eval_dataset, test_dataset)
    y_true, feat_idx, feat_val = batch_data

    params = {'lr': args.learning_rate,
              'l2_reg': args.l2_reg,
              'dropout_rate': args.dropout_rate,
              'batch_size': args.batch_size,
              'emb_size': args.emb_size}
    logging.info('params: {}'.format(params))
    model_module = import_module('models.{}'.format(args.model_name))
    model = model_module.Model(vocab_size, args.field_num, params)
    model.init_graph(y_true, feat_idx, feat_val, handle)

    session_conf = tf.ConfigProto()
    session_conf.gpu_options.allow_growth = True
    with tf.Session(config=session_conf) as sess:
        sess.run(tf.global_variables_initializer())
        logging.info('----model running----')
        train_and_eval(
            train_data_iter, val_data_iter, eval_data_iter, model, sess, args.eval_step, args.model_dir)
        ckpt = tf.train.latest_checkpoint(args.model_dir)
        model.saver.restore(sess, ckpt)
        test_auc, test_logloss = evaluate(test_data_iter, model, sess)
        logging.info('test auc: {:.4}, test logloss: {:.4}'.format(test_auc, test_logloss))
        logging.info('----done----')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_size_file',
                        type=str,
                        required=True,
                        help='feature size info file')
    parser.add_argument('--train_data',
                        type=str,
                        required=True,
                        help='train data path')
    parser.add_argument('--val_data',
                        type=str,
                        required=True,
                        help='val data path(select from train data)')
    parser.add_argument('--eval_data',
                        type=str,
                        required=True,
                        help='eval data path')
    parser.add_argument('--test_data',
                        type=str,
                        required=True,
                        help='test data path')
    parser.add_argument('--model_name',
                        type=str,
                        required=True,
                        help='model name')
    parser.add_argument('--model_dir',
                        type=str,
                        required=True,
                        help='save model checkpoint directory')
    parser.add_argument('--learning_rate', '-lr',
                        type=float,
                        required=True)
    parser.add_argument('--field_num',
                        type=int,
                        required=True)
    parser.add_argument('--epoch',
                        type=int,
                        required=True)
    parser.add_argument('--l2_reg',
                        type=float,
                        required=True)
    parser.add_argument('--dropout_rate',
                        type=float,
                        required=True)
    parser.add_argument('--batch_size',
                        type=int,
                        required=True)
    parser.add_argument('--emb_size',
                        type=int,
                        required=True)
    parser.add_argument('--eval_step',
                        type=int,
                        required=True)
    parser.add_argument('--version',
                        type=str,
                        required=True)
    parser.add_argument('--gpu',
                        type=str,
                        required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    tf.logging.set_verbosity(tf.logging.INFO)
    logging.basicConfig(level=logging.INFO,
                        filename='./log_{}.txt'.format(args.version),
                        filemode='w',
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    with tf.Graph().as_default():
        run(args)
