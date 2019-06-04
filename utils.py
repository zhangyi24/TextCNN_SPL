#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
	Author: Yi Zhang
	Date created: 2019/06/04
'''
import yaml
import pickle

from tensorflow.contrib.training import HParams


def load_config(yaml_file, section):
	with open(yaml_file, 'r', encoding='utf-8') as f:
		cfg = yaml.safe_load(f)[section]
	if section == 'hparams':
		hparams = HParams(**cfg)
		return hparams
	else:
		return cfg


def load_vocab(vocab_path):
	with open(vocab_path, 'rb') as f:
		return pickle.load(f)


def load_embed(embed_path):
	with open(embed_path, 'rb') as f:
		return pickle.load(f)


if __name__ == '__main__':
	cfg_file= 'config.yaml'
	files_path = load_config(cfg_file, section='path')
	hp = load_config(cfg_file, section='hparams')
	





