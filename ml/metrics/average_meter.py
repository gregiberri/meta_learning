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


class SceneMeter(object):
    def __init__(self):
        self.scenes = dict()

    def reset(self):
        self.scenes = dict()

    def add_scene(self, scene):
        self.scenes[scene] = AverageMeterList()

    def update(self, x, scene):
        if not scene in self.scenes.keys():
            self.add_scene(scene)

        self.scenes[scene].update(x)

    def mean(self):
        return {key: scene_meter.mean() for key, scene_meter in self.scenes.items()}

    def values(self):
        return {key: scene_meter.mean() for key, scene_meter in self.scenes.items()}

    def sort_by_scene_name(self):
        self.scenes = collections.OrderedDict(sorted(self.scenes.items()))

    def draw_histogram(self, path):
        self.sort_by_scene_name()
        means = self.mean()
        plt.bar(means.keys(), means.values())
        # plt.ylim([0, 3500])
        plt.savefig(path)
        plt.close()


class AverageMeterDict(object):
    def __init__(self):
        self.data = None
        self.count = 0

    def reset(self):
        del self.data
        self.data = None
        self.count = 0

    def update(self, x):
        check_allfloat(x)
        self.count += 1
        if self.data is None:
            self.data = copy.deepcopy(x)
        else:
            # TODO: make it iteration for any type dict.
            for k1, v1 in x.items():
                if isinstance(v1, float):
                    self.data[k1] += v1
                elif isinstance(v1, tuple) or isinstance(v1, list):
                    for idx, v2 in enumerate(v1):
                        self.data[k1][idx] += v2
                else:
                    assert NotImplementedError("error input type for update AvgMeterDict")

    def mean(self):
        @make_iterative_func
        def get_mean(v):
            return v / float(self.count)

        return get_mean(self.data)
