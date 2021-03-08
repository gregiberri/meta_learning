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
from ml.metrics.average_meter import AverageMeterList, SceneMeter

matplotlib.use('Agg')


def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return torch.log(x) / math.log(10)


@make_nograd_func
def compute_metric(output, target):
    # print("pred shape: {}, target shape: {}".format(pred.shape, target.shape))
    assert output.shape == target.shape, "pred'shape must be same with target."
    valid_mask = target > 0
    output = output[valid_mask]
    target = target[valid_mask]

    abs_diff = (output - target).abs()

    mse = (torch.pow(abs_diff, 2)).mean()
    rmse = torch.sqrt(mse) * 1000
    mae = abs_diff.mean()
    absrel = (abs_diff / target).mean()

    lg10 = torch.abs(log10(output) - log10(target)).mean()

    err_log = torch.log(target) - torch.log(output)
    normalized_squared_log = (err_log ** 2).mean()
    log_mean = err_log.mean()
    silog = torch.sqrt(normalized_squared_log - log_mean * log_mean) * 100

    maxRatio = torch.max(output / target, target / output)
    delta1 = (maxRatio < 1.25).float().mean()
    delta2 = (maxRatio < 1.25 ** 2).float().mean()
    delta3 = (maxRatio < 1.25 ** 3).float().mean()

    # inv_output = 1 / pred
    # inv_target = 1 / target
    # abs_inv_diff = (inv_output - inv_target).abs()
    # irmse = torch.sqrt((torch.pow(abs_inv_diff, 2)).mean())
    # imae = abs_inv_diff.mean()

    inv_output_km = (1e-3 * output) ** (-1)
    inv_target_km = (1e-3 * target) ** (-1)
    abs_inv_diff = (inv_output_km - inv_target_km).abs()
    irmse = torch.sqrt((torch.pow(abs_inv_diff, 2)).mean())
    imae = abs_inv_diff.mean()

    metric = dict(mse=mse,
                  rmse=rmse,
                  mae=mae,
                  absrel=absrel,
                  lg10=lg10,
                  silog=silog,
                  delta1=delta1,
                  delta2=delta2,
                  delta3=delta3,
                  irmse=irmse,
                  imae=imae)

    return metric


class Metrics(object):

    def __init__(self, snap_dir='', tag='train', niter=None):
        self.epoch = 0

        self.irmse = AverageMeterList()
        self.imae = AverageMeterList()

        self.mse = AverageMeterList()
        self.mae = AverageMeterList()
        self.rmse = AverageMeterList()
        self.absrel = AverageMeterList()

        self.silog = AverageMeterList()

        self.d1 = AverageMeterList()
        self.d2 = AverageMeterList()
        self.d3 = AverageMeterList()

        self.result_names = list(self.__dict__.keys())
        self.epoch_results = {result_name: [] for result_name in self.result_names if result_name != 'epoch'}

        self.rmse_scene_meter = SceneMeter()

        self.snap_dir = snap_dir
        self.tag = tag
        self.niter = niter
        self.make_metrics_file()

        self.n_stage = -1

    def reset(self):
        self.irmse.reset()
        self.imae.reset()

        self.mse.reset()
        self.mae.reset()
        self.rmse.reset()
        self.absrel.reset()

        # self.lg10.reset()
        self.silog.reset()

        self.d1.reset()
        self.d2.reset()
        self.d3.reset()

        self.rmse_scene_meter.reset()

        self.n_stage = -1

    def compute_metric(self, preds, minibatch, scene='scene'):
        if minibatch["target"].shape != preds.shape:
            h, w = minibatch["target"].shape[-2:]
            # minibatch = interpolate(minibatch, size=(h, w), mode='bilinear', align_corners=True)
            preds = interpolate(preds, size=(h, w), mode='bilinear', align_corners=True)
        gt = minibatch["target"]
        mask = (gt > 0.)
        if len(gt[mask]) == 0:
            return

        if self.n_stage == -1:
            self.n_stage = len(preds)

        for scale_idx in range(self.n_stage):
            pred = preds[scale_idx]
            gt = minibatch["target"][scale_idx]
            metirc_dict = compute_metric(pred, gt)
            # TODO what is it: # metirc_dict = reduce_dict(metirc_dict)
            self.irmse.update(metirc_dict["irmse"].cpu())
            self.imae.update(metirc_dict["imae"].cpu())
            self.mse.update(metirc_dict["mse"].cpu())
            self.mae.update(metirc_dict["mae"].cpu())
            self.rmse.update(metirc_dict["rmse"].cpu())
            self.absrel.update(metirc_dict["absrel"].cpu())
            self.silog.update(metirc_dict["silog"].cpu())
            self.d1.update(metirc_dict["delta1"].cpu())
            self.d2.update(metirc_dict["delta2"].cpu())
            self.d3.update(metirc_dict["delta3"].cpu())
            self.rmse_scene_meter.update(metirc_dict["rmse"].cpu(), scene[scale_idx])

    def add_scalar(self, writer=None, iteration=0):
        if writer is None:
            return
        for key in self.result_names:
            if key != 'epoch':
                writer.add_scalar(self.tag + f'/{key}', self.__dict__[key].mean(), self.epoch * self.niter + iteration)

    def get_snapshot_info(self):
        info = "absrel: %.2f" % self.absrel.values() + "(%.2f)" % self.absrel.mean()
        info += " rmse: %.2f" % self.rmse.values() + "(%.2f)" % self.rmse.mean()
        info += " irmse: %.2f" % self.irmse.values() + "(%.2f)" % self.irmse.mean()
        info += " silog: %.2f" % self.silog.values() + "(%.2f)" % self.silog.mean()
        return info

    def make_metrics_file(self):
        file_path = os.path.join(self.snap_dir, f'{self.tag}_results.csv')
        if not os.path.exists(file_path):
            with open(file_path, mode='w') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self.result_names)
                writer.writeheader()

    def save_metrics(self):
        # make result dict
        result_dict = self.get_result_means_dict()
        with open(os.path.join(self.snap_dir, f'{self.tag}_results.csv'), mode='a+') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.result_names)
            writer.writerow(result_dict)

    def get_result_values_dict(self):
        result_dict = dict()
        for key in self.result_names:
            if key == 'epoch':
                result_dict[key] = self.__dict__[key]
            else:
                result_dict[key] = self.__dict__[key].values()
        return result_dict

    def get_result_means_dict(self):
        result_dict = dict()
        for key in self.result_names:
            if key == 'epoch':
                result_dict[key] = self.__dict__[key]
            else:
                result_dict[key] = self.__dict__[key].mean()
        return result_dict

    def get_result_info(self):
        info = "absrel: %.2f" % self.absrel.mean() + \
               " rmse: %.2f" % self.rmse.mean() + \
               " irmse: %.2f" % self.irmse.mean() + \
               " silog: %.2f" % self.silog.mean()
        return info

    def on_epoch_end(self):
        self.save_metrics()
        self.append_to_epoch_results()
        self.rmse_scene_meter.draw_histogram(os.path.join(self.snap_dir, f'{self.tag}_epoch_{self.epoch}.png'))
        self.reset()
        self.epoch += 1

    def append_to_epoch_results(self):
        results_dict = self.get_result_means_dict()
        for key, value in results_dict.items():
            if key != 'epoch':
                self.epoch_results[key].append(value)
