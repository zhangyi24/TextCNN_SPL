import csv
import yaml
import random
import pickle
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.training import HParams
import jieba


def load_config(yaml_file, section):
	with open(yaml_file, 'r', encoding='utf-8') as f:
		cfg = yaml.load(f)
	return cfg[section]


def load_hparams(yaml_file, section):
	hparams = load_config(yaml_file, section)
	hparams = HParams(**hparams)
	return hparams

	

def load_vocab(vocab_path):
	with open(vocab_path, 'rb') as f:
		return pickle.load(f)


def load_embed(embed_path):
	with open(embed_path, 'rb') as f:
		return pickle.load(f)


if __name__ == '__main__':
	cfg_path = 'config.yaml'
	dataset_name = load_config(cfg_path, section='default')['dataset']
	cfg = load_config(cfg_path, section=dataset_name)
	hparams_path = 'config/hparams.yaml'
	hp = load_hparams(hparams_path, section='default')
	





