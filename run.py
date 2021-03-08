#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2019-12-24 22:54
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : train.py
"""

import argparse
import logging
import warnings


warnings.filterwarnings("ignore")
logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

from ml.solver import Solver

parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('--id', type=str, default='debug', help='Id of the training')
parser.add_argument('--mode', type=str, default='train', choices=['pretrain', 'train', 'val', 'resume', 'hyperopt'],
                    help='The mode of the running.')
parser.add_argument('-c', '--config', type=str, default='config/config_files/mini_imagenet_base.yaml', help='Config file name')

args = parser.parse_args()

if __name__ == '__main__':
    solver = Solver(args)
    solver.run()
