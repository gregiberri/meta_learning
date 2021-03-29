# -*- coding: utf-8 -*-
# @Time    : 2019/12/26 下午4:09
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com
# @File    : average_meter.py
import collections
import copy
import numpy as np
from ml.utils.wrappers import make_iterative_func
from ml.utils.pyt_ops import check_allfloat
import matplotlib.pyplot as plt


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def mean(self):
        self.count = 1 if self.count == 0 else self.count
        return self.sum / self.count

    def values(self):
        return self.val


class AverageMeterList(object):
    def __init__(self):
        self.vals = None
        self.sums = None
        self.count = 0

    def reset(self):
        self.vals = None
        self.sums = None
        self.count = 0

    def update(self, x):
        self.count += 1
        if self.vals is None:
            self.vals = np.array(x).astype(np.float32)
            self.sums = np.array(x).astype(np.float32)
        else:
            self.vals = np.array(x).astype(np.float32)
            self.sums += self.vals

    def mean(self):
        return (self.sums / self.count).tolist()

    def values(self):
        return self.vals.tolist()
