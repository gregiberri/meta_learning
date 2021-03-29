#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Author  : Albert Gregus
@Email   : albert.gregus@continental-corporation.com
"""

import argparse
import logging
import warnings

from config import ConfigNamespace
from ml.solvers import get_solver

warnings.filterwarnings("ignore")
logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

from ml.solvers.base_solver import Solver

parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('--id', type=str, default='debug', help='Id of the training')
parser.add_argument('--mode', type=str, default='train', choices=['pretrain', 'train', 'val', 'resume', 'hyperopt'],
                    help='The mode of the running.')
parser.add_argument('-c', '--config', type=str, default='config/config_files/mini_imagenet_base.yaml',
                    help='Config file name')
parser.add_argument('--resume', action='store_true', help='Set to resume the training.')

args = parser.parse_args()

if __name__ == '__main__':
    config = ConfigNamespace(args.config)
    solver = get_solver(config, args)
    solver.run()
