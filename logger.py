#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
    Author: Yi Zhang
    Date created: 2019/06/04
'''

import logging
import sys
import os
import datetime


def create_logger(name):
	logger = logging.getLogger(name=name)
	logger.setLevel(logging.DEBUG)
	fmt = logging.Formatter('%(asctime)s %(message)s')
	logfile = get_log_filename()
	fh = logging.FileHandler(filename=logfile, mode='w', encoding='utf-8')
	fh.setLevel(logging.DEBUG)
	fh.setFormatter(fmt)
	sh = logging.StreamHandler(stream=sys.stdout)
	sh.setLevel(logging.DEBUG)
	sh.setFormatter(fmt)
	logger.addHandler(fh)
	logger.addHandler(sh)
	return logger


def get_log_filename():
	file_dir = create_or_get_dir()
	time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
	filename = time + '.log'
	filename = os.path.join(file_dir, filename)
	return filename


def create_or_get_dir():
	directory = 'log'
	if not os.path.exists(directory):
		os.mkdir(directory)
	return directory


if __name__ == '__main__':
	logger = create_logger('test')
	logger.info('1')
	logger.info('2')
	logger.info('3')