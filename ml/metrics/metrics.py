# -*- coding: utf-8 -*-
# @Time    : 2020/1/7 下午4:24
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com
# @File    : metrics.py
import csv
import math
import os
import matplotlib
import torch

from ml.utils.wrappers import make_nograd_func
from ml.utils.pyt_ops import interpolate
from ml.utils.comm import reduce_dict
from ml.metrics.average_meter import AverageMeterList

matplotlib.use('Agg')


@make_nograd_func
def compute_metric(output, target, loss):
    accuracy = torch.mean((output == target).float())

    metric = dict(accuracy=accuracy,
                  loss=loss)

    return metric


class Metrics(object):

    def __init__(self, save_dir='', tag='train'):
        self.acc = AverageMeterList()
        self.loss = AverageMeterList()

        self.result_names = ['epoch'] + list(self.__dict__.keys())
        self.epoch_results = {result_name: [] for result_name in self.result_names if result_name != 'epoch'}

        self.save_dir = save_dir
        self.tag = tag
        self.make_metrics_file()

        self.n_stage = -1

    def reset(self):
        self.acc.reset()
        self.loss.reset()

    def compute_metric(self, pred_dict):
        self.acc.update(pred_dict["accuracy"].detach().cpu())
        self.loss.update(pred_dict["loss"].detach().cpu())

    def add_scalar(self, writer=None, iteration=0, epoch=0):
        #todo
        if writer is None:
            return
        for key in self.result_names:
            if key != 'epoch':
                writer.add_scalar(self.tag + f'/{key}', self.__dict__[key].mean(), epoch * self.niter + iteration)

    def get_snapshot_info(self):
        info = "loss: %.3f" % self.loss.values() + "(%.3f)" % self.loss.mean()
        info += " accuracy: %.3f" % self.acc.values() + "(%.3f)" % self.acc.mean()
        info += ' '
        return info

    def make_metrics_file(self):
        file_path = os.path.join(self.save_dir, f'{self.tag}_results.csv')
        if not os.path.exists(file_path):
            with open(file_path, mode='w') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self.result_names)
                writer.writeheader()

    def save_metrics(self, epoch):
        # make result dict
        result_dict = self.get_result_means_dict(epoch)
        with open(os.path.join(self.save_dir, f'{self.tag}_results.csv'), mode='a+') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.result_names)
            writer.writerow(result_dict)

    def get_result_values_dict(self, epoch):
        result_dict = dict()
        result_dict['epoch'] = epoch
        for key in self.result_names:
            if key != 'epoch':
                result_dict[key] = self.__dict__[key].values()
        return result_dict

    def get_result_means_dict(self, epoch):
        result_dict = dict()
        result_dict['epoch'] = epoch
        for key in self.result_names:
            if key != 'epoch':
                result_dict[key] = self.__dict__[key].mean()
        return result_dict

    def get_result_info(self):
        info = "accuracy: %.2f" % self.acc.mean()
        return info

    def on_epoch_end(self, epoch):
        self.save_metrics(epoch)
        self.append_to_epoch_results(epoch)
        self.reset()

    def append_to_epoch_results(self, epoch):
        results_dict = self.get_result_means_dict(epoch)
        for key, value in results_dict.items():
            if key != 'epoch':
                self.epoch_results[key].append(value)
