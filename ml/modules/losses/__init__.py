#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020-05-03 04:35
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : __init__.py
"""

import torch.nn
from ml.modules.losses.masked_l1_loss import MaskedL1Loss, MaskedL2Loss


def get_regression_loss(regression_loss):
    if regression_loss == 'L1':
        return MaskedL1Loss()
    elif regression_loss == 'L2':
        return MaskedL2Loss()
    else:
        raise ValueError(f'Wrong regression loss: {regression_loss}')


def get_classification_loss(classification_loss):
    if classification_loss.name == 'cross_entropy':
        return torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f'Wrong classification loss: {classification_loss}')
