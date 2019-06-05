#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
    Author: Yi Zhang
    Date created: 2019/06/04
'''
from collections import Counter
import pickle
import random

import jieba
import numpy as np

from utils import load_config
from logger import create_logger


UNK = '<UNK>'
PAD = '<PAD>'

class Dataset(object):
    def __init__(self, data_path, logger, dict_class_to_label=None):
        self._data_path = data_path
        self._logger = logger
        self.data, self.classes_stat = read_data(self._data_path)
        self._data_size = len(self.data)
        self.num_classes = len(self.classes_stat)
        self.dict_class_to_label = dict_class_to_label
        self.dict_class_to_label, self.dict_label_to_class = self.link_label_and_class()
        self._generate_label()  # get label from class_id, label range from [0, 1, ..., num_classes - 1]
        self.label_stat = self._get_label_stat()
        self.len_stat = self._get_sents_len_stat()  # statistic of length of sentences
        self._next_batch_start = 0

    def link_label_and_class(self):
        if self.dict_class_to_label:
            dict_class_to_label = self.dict_class_to_label
        else:
            labels = list(self.classes_stat.keys())
            labels.sort()
            dict_class_to_label = dict(zip(labels, range(len(labels))))
        dict_label_to_class = dict(zip(dict_class_to_label.values(), dict_class_to_label.keys()))
        return dict_class_to_label, dict_label_to_class

    def next_batch(self, batch_size=None):
        if batch_size:
            if self._next_batch_start == 0:
                random.shuffle(self.data)
            start = self._next_batch_start
            end = min(self._next_batch_start + batch_size, len(self.data))
            batch_data = self.data[start: end]
            if end == self._data_size:
                self._next_batch_start = 0
            else:
                self._next_batch_start = end
            return batch_data
        else:
            return self.data

    def _generate_label(self):
        for datum in self.data:
            try:
                label = self.dict_class_to_label[datum['class_id']]
                datum.update({'label': label})
            except KeyError:
                self._logger.info('KeyError: class_id: %s is not in dict_class_to_label: %s', datum['class_id'], self.dict_class_to_label)

    def _get_label_stat(self):
        label_stat = {}
        for class_id, num in self.classes_stat.items():
            label = self.dict_class_to_label[class_id]
            label_stat.update({label: num})
        return label_stat

    def _get_sents_len_stat(self):
        len_stat = {}
        sents_len = []
        for datum in self.data:
            sents_len.append(len(jieba.lcut(datum['sent'])))
        len_stat.update({'min': np.min(sents_len), 'max': np.max(sents_len), 'median': np.median(sents_len)})
        cnt = Counter(sents_len)
        cnt = list(cnt.items())
        cnt.sort(key=lambda x: x[0])
        len_stat.update({'cnt': cnt})
        return len_stat

    def __len__(self):
        return self._data_size


def read_data(data_path):
    dataset = []
    classes_stat = {}
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for data_id, line in enumerate(lines):
            datum = line.split(',')
            sent = datum[1]
            words = jieba.lcut(sent)
            class_id = int(datum[5])
            dataset.append({'sent': sent, 'words': words, 'class_id': class_id, 'data_id': data_id})
            if class_id in classes_stat:
                classes_stat[class_id] += 1
            else:
                classes_stat[class_id] = 1
    return dataset, classes_stat


def sent_to_words_id(batch_sent, max_len, vocab):
    batch_size = len(batch_sent)
    batch_words_id = np.zeros(shape=[batch_size, max_len], dtype=np.int32)
    for sent_id, sent in enumerate(batch_sent):
        len_sent = min(len(sent), max_len)
        sent = sent[0: len_sent]
        for word_id, word in enumerate(sent):
            batch_words_id[sent_id, word_id] = vocab[word] if word in vocab else vocab[UNK]
        batch_words_id[sent_id, len_sent: max_len] = vocab[PAD]
    return batch_words_id


def data_to_num(batch_data, max_len, vocab, embed, num_classes):
    batch_sent = [datum['words'] for datum in batch_data]
    batch_words_id = sent_to_words_id(batch_sent, max_len, vocab)
    embed_sent = embed[batch_words_id]

    label = [datum['label'] for datum in batch_data]
    onehot_label = np.eye(num_classes, dtype=np.int32)[label]
    return embed_sent, onehot_label


def feed_data(model, embed_sents, onehot_labels, training):
	feed_dict = {
		model.embed_sents: embed_sents,
		model.onehot_labels: onehot_labels,
        model.batch_data_len: len(onehot_labels),
		model.training: training
	}
	return feed_dict


def load_vocab(file):
    with open(file, 'r', encoding='utf-8') as f:
        words = f.read().split()
    vocab = dict(zip(words, range(len(words))))
    return vocab


def load_embed(file):
    with open(file, 'rb') as f:
        embed = pickle.load(f, encoding='utf-8')
        return embed

if __name__ == '__main__':
    files_path = load_config('config.yaml', 'path')
    logger = create_logger('data')

    vocab = load_vocab(files_path['vocab_path'])
    embed = load_embed(files_path['embed_path'])
    trainset = Dataset(data_path=files_path['train_data_path'], logger=logger)
    hp = load_config('config.yaml', 'hparams')
    print(trainset.next_batch(2))



