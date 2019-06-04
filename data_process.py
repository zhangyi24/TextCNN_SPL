#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
    Author: Yi Zhang
    Date created: 2019/06/04
'''
from collections import Counter

import jieba
import numpy as np

from utils import load_config
from logger import create_logger


class Dataset(object):
    def __init__(self, data_path, logger, label_table=None):
        self._data_path = data_path
        self._logger = logger
        self.data, self.classes_stat = read_data(self._data_path)
        self._data_size = len(self.data)
        self.num_classes = len(self.classes_stat)
        if label_table:
            self.label_table = label_table
        else:
            self.label_table = self.create_label_table()    # map class_id to label
        self._class_to_label()  # get label from class_id, label range from [0, 1, ..., num_classes - 1]
        self.len_stat = self._get_sents_len_stat()  # statistic of length of sentences
        self._next_batch_start = 0

    def create_label_table(self):
        labels = list(self.classes_stat.keys())
        labels.sort()
        label_table = dict(zip(labels, range(len(labels))))
        self._logger.info('label_table:%s', label_table)
        return label_table

    def next_batch(self, batch_size=None):
        if batch_size:
            start = self._next_batch_start
            end = max(self._next_batch_start + batch_size, len(self.data))
            batch_data = self.data[start: end]
            if end == self._data_size:
                self._next_batch_start = 0
            else:
                self._next_batch_start = end
            return batch_data

    def _class_to_label(self):
        for datum in self.data:
            try:
                label = self.label_table[datum['class_id']]
                datum.update({'label': label})
            except KeyError:
                self._logger.info('KeyError: class_id: %s is not in label_table: %s', datum['class_id'], self.label_table)

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

def sent2num(sent, vocab, embed):
    words = jieba.lcut(sent)
    for word in words:
        vocab

def data2num(batch_data, vocab, embed):
    emded_sents

def read_data(data_path):
    dataset = []
    classes_stat = {}
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            datum = line.split(',')
            sent = datum[1]
            class_id = int(datum[5])
            dataset.append({'sent': sent, 'class_id': class_id})
            if class_id in classes_stat:
                classes_stat[class_id] += 1
            else:
                classes_stat[class_id] = 1
    return dataset, classes_stat


def feed_data(model, embed_sen, onehot_labels, training):
	feed_dict = {
		model.embed_sen: embed_sen,
		model.onehot_labels: onehot_labels,
		model.training: training
	}
	return feed_dict

if __name__ == '__main__':
    files_path = load_config('config.yaml', 'path')
    logger = create_logger('data')

    train_data_path = files_path['train_data_path']
    trainset = Dataset(train_data_path, logger)
    valid_data_path = files_path['valid_data_path']
    validset = Dataset(valid_data_path, logger, label_table=trainset.label_table)




