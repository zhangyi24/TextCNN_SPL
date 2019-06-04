#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
	Author: Yi Zhang
	Date created: 2019/06/04
'''
import os

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np

from data_process import Dataset, feed_data, load_vocab, load_embed, data_to_num
from utils import load_config
from model import TextCNN
from logger import create_logger

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_boolean('debug', default=False, help='use tfdbg for debugging')


def evaluate(sess, model, dataset, vocab, embed):
	batch_size = model.hp.batch_size_eval
	num_batch = (len(dataset) - 1) // batch_size + 1
	num_samples, loss_total, correct_total = 0, 0, 0
	for id_batch in range(num_batch):
		batch_data = dataset.next_batch(batch_size)
		len_batch_data = len(batch_data)
		batch_data = data_to_num(batch_data,
								 max_len=model.hp.max_len,
								 vocab=vocab,
								 embed=embed,
								 num_classes = dataset.num_classes)
		feed_dict = feed_data(model, *batch_data, training=False)
		loss_batch, correct_batch = sess.run(fetches=(model.loss, model.correct), feed_dict=feed_dict)
		num_samples += len_batch_data
		loss_total += loss_batch * len_batch_data
		correct_total += np.sum(correct_batch)
	loss = loss_total / num_samples
	acc = correct_total / num_samples
	return loss, acc


def train_epoch(sess, model, trainset, vocab, embed, id_epoch):
	batch_size = model.hp.batch_size_train
	num_batch = (len(trainset) - 1) // batch_size + 1
	num_samples, loss_total, correct_total = 0, 0, 0
	for id_batch in range(num_batch):
		batch_data = trainset.next_batch(batch_size)
		len_batch_data = len(batch_data)
		batch_data = data_to_num(batch_data=batch_data,
								 max_len=model.hp.max_len,
								 vocab=vocab,
								 embed=embed,
								 num_classes=trainset.num_classes)
		feed_dict = feed_data(model, *batch_data, training=True)
		if FLAGS.debug:
			loss_batch, correct_batch, learning_rate, global_step, _, _ = sess.run(
				fetches=(model.loss, model.correct, model.learning_rate, model.global_step, model.train, model.print), feed_dict=feed_dict)
		else:
			loss_batch, correct_batch, learning_rate, global_step, _ = sess.run(
				fetches=(model.loss, model.correct, model.learning_rate, model.global_step, model.train), feed_dict=feed_dict)
		num_samples += len_batch_data
		loss_total += loss_batch * len_batch_data
		correct_total += np.sum(correct_batch)
		if (id_batch + 1) % 20 == 0:
			loss_train = loss_total / num_samples
			acc_train = correct_total / num_samples
			model.logger.info('Epoch: %d\tbatch: %d\tloss_train: %.4f\tacc_train: %.4f\tlr: %.4e\tstep: %d',
				id_epoch + 1, id_batch+1, loss_train, acc_train, learning_rate, global_step)
			num_samples, loss_total, correct_total = 0, 0, 0


def main(_):
	cfg_file = 'config.yaml'
	hparams = load_config(cfg_file, section='hparams')		# hparams
	logger = create_logger('textcnn')		# logger

	# prepare dataset, vocab, embed
	files_path = load_config(cfg_file, section='path')
	trainset = Dataset(files_path['train_data_path'], logger)
	num_classes = trainset.num_classes
	validset = Dataset(files_path['train_data_path'], logger, label_table=trainset.label_table)
	vocab = load_vocab(files_path['vocab_path'])
	word_embed = load_embed(files_path['word_embed_path'])

	logger.info('loading model...')
	graph = tf.Graph()
	with graph.as_default():
		model = TextCNN(hparams=hparams, num_classes=num_classes, logger=logger)

	with tf.Session(graph=graph) as sess:
		# debug
		if FLAGS.debug:
			sess = tf_debug.LocalCLIDebugWrapperSession(sess, dump_root='tfdbg')
		# init model
		sess.run(tf.global_variables_initializer())
		logger.info('params initialized')

		# create a saver
		saver = tf.train.Saver()
		save_path = files_path['save_model_path']
		os.makedirs(os.path.dirname(save_path), exist_ok=True)

		# performance of the model before training
		loss_valid, acc = evaluate(sess, model, validset, vocab, word_embed)
		logger.info('loss_valid: %.4f\tacc: %.4f', loss_valid, acc)
		best_result = {'loss_valid': loss_valid, 'acc': acc}
		patience = 0
		# train
		for id_epoch in range(hparams.num_epoch):
			train_epoch(sess, model, trainset, vocab, word_embed, id_epoch)
			loss_valid, acc = evaluate(sess, model, validset, vocab, word_embed)
			logger.info('Epoch: %d\tloss_valid: %.4f\tacc: %.4f', id_epoch + 1, loss_valid, acc)
			if loss_valid < best_result['loss_valid']:		# save model
				saver.save(sess=sess, save_path=save_path)
				logger.info('model saved in %s', save_path)
				best_result = {'loss_valid': loss_valid, 'acc': acc}
				patience = 0
			else:		# early stopping
				patience += 1
				if patience >= hparams.earlystop_patience:
					logger.info('earlystop.')
					logger.info('Best result: loss_valid: %.4f\tacc: %.4f',
								best_result['loss_valid'], best_result['acc'])
					break

if __name__ == '__main__':
	tf.app.run()
