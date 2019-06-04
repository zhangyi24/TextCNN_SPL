#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
    Author: Yi Zhang
    Date created: 2019/06/04
'''
import sys
import copy

import tensorflow as tf

from utils import load_config, load_hparams
from logger import create_logger


class TextCNN(object):
	def __init__(self, hparams, num_classes, logger):
		# hparams
		self._hp = copy.deepcopy(hparams)
		self._num_classes = num_classes
		self._logger = logger
		# placeholder
		self.embed_sent = tf.placeholder(dtype=tf.float32, shape=[None, self._hp.max_len, self._hp.embed_size],
										  name='embed_sents')
		self.onehot_label = tf.placeholder(dtype=tf.int32, shape=[None, num_classes], name='Onehot_label')
		self.training = tf.placeholder(dtype=tf.bool, shape=None, name='training')
		
		# TextCNN model
		self._initializer = tf.initializers.glorot_normal()
		with tf.variable_scope('TextCNN'):
			with tf.variable_scope('CNN'):
				self.features = cnn(input_tensor=self.embed_sent,
									num_filters=self._hp.num_filters,
									filter_sizes=self._hp.filter_sizes,
									initializer=self._initializer)
			with tf.variable_scope('FC'):
				self._logits = fully_connected_layer(input_tensor=self.features,
													units=num_classes,
													dropout_rate=self._hp.dropout_rate,
													training=self.training,
													initializer=self._initializer)
		# calculate loss, pred, prob, correct, acc based on logits
		self._tvars = tf.trainable_variables()
		with tf.name_scope('loss'):
			self._loss = self._calc_loss()
		with tf.name_scope('probability'):
			self._prob = tf.nn.softmax(self._logits, axis=1, name='Prob')
		with tf.name_scope('prediction'):
			self._pred = tf.argmax(self._logits, axis=1, output_type=tf.int32, name='Pred')
		with tf.name_scope('is_correct'):
			self._label = tf.argmax(self.onehot_label, axis=1, output_type=tf.int32, name='Label')
			self._correct = tf.equal(self._pred, self._label, name='Correct')
		
		self.train = self._train_op()
		self.print = self._print_op()
		self._print_hparams()
		self._print_tvars()

	# todo: 改成SPL
	def _calc_loss(self):
		loss = tf.losses.softmax_cross_entropy(onehot_labels=self.onehot_label, logits=self._logits,
		                                             label_smoothing=self._hp.label_smoothing)
		if self._hp.scale_l2:
			loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in self._tvars if 'bias' not in v.name])
			loss = loss + self._hp.scale_l2 * loss_l2
		return loss

	def _train_op(self):
		self._global_step = tf.train.create_global_step()
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self._hp.learning_rate)
		gvs = self.optimizer.compute_gradients(self._loss)
		g, v = zip(*gvs)
		if self._hp.grad_max:
			g, _ = tf.clip_by_global_norm(g, self._hp.grad_max)
		clipped_gvs = zip(g, v)
		return self.optimizer.apply_gradients(clipped_gvs, global_step=self._global_step)

	def _print_op(self):
		# for debug
		tensors_to_print = [self._tvars]
		print_op = tf.print(*tensors_to_print, output_stream=sys.stdout)
		return print_op

	def _print_hparams(self):
		self._logger.info('hyperparameters:')
		for k, v in self._hp.values().items():
			self._logger.info('\t%s: %s', k, v)

	def _print_tvars(self):
		self._logger.info('trainable variables:')
		for v in self._tvars:
			self._logger.info('\t%s', v)
	
	@property
	def hp(self):
		return self._hp
	
	@property
	def loss(self):
		return self._loss
	
	@property
	def pred(self):
		return self._pred

	@property
	def correct(self):
		return self._correct
	
	@property
	def global_step(self):
		return self._global_step


def cnn(input_tensor, num_filters, filter_sizes, initializer):
	# [B, W, C] -> [B, 1, W, C]
	input_tensor = tf.expand_dims(input_tensor, axis=1, name='3D_to_4D')
	input_shape = input_tensor.shape
	sent_len = input_shape[2]
	embed_size = input_shape[3]
	output = []
	for i, filter_size in enumerate(filter_sizes):
		with tf.variable_scope('filter_size_%d' % filter_size):
			with tf.variable_scope('conv'):
				# [H_k, W_k, Cin, Cout]
				kernel = tf.get_variable(name='filter',
										 shape=[1, filter_size, embed_size, num_filters],
										 dtype=tf.float32,
										 initializer=initializer,
										 trainable=True)
				bias = tf.get_variable(name='bias',
									   shape=[num_filters],
									   dtype=tf.float32,
									   initializer=tf.zeros_initializer(),
									   trainable=True)
				conv = tf.nn.conv2d(input=input_tensor, filter=kernel, strides=[1, 1, 1, 1], padding='VALID')
				conv = tf.nn.bias_add(conv, bias)
				conv = tf.nn.relu(conv)
			with tf.variable_scope('pooling'):
				pooled = tf.nn.max_pool(value=conv,
										ksize=[1, 1, sent_len - filter_size + 1, 1],
										strides=[1, 1, 1, 1],
										padding='VALID')
				pooled = tf.squeeze(pooled, axis=[1, 2])
			output.append(pooled)
	output = tf.concat(output, axis=1)
	return output


def fully_connected_layer(input_tensor, units, dropout_rate, training, initializer):
	output = tf.layers.dropout(input_tensor, dropout_rate, training=training)
	output = tf.layers.dense(output, units=units, activation=tf.nn.relu, kernel_initializer=initializer)
	return output


if __name__ == '__main__':
	cfg_file = 'config.yaml'
	files_path = load_config(cfg_file, section='path')
	hparams = load_hparams(cfg_file, section='hparams')
	logger = create_logger('TextCNN')
	model = TextCNN(hparams=hparams, num_classes=9, logger=logger)
