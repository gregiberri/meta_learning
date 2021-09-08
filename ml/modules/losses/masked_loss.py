#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020-05-04 23:07
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : masked_loss.py
"""
import torch

from ml.utils.pyt_ops import interpolate


class MaskedL1Loss(object):

    def __init__(self):
        super().__init__()

    def __call__(self, pred, gt):
        assert pred.dim() == gt.dim(), \
            "inconsistent dimensions, pred shape is {}, but gt shape is {}.".format(pred.shape, gt.shape)

        if pred.shape != gt.shape:
            pred = interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=True)

        valid_mask = (gt != 0).detach()
        diff = gt - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().sum()
        return self.loss


class MaskedL2Loss(object):

    def __init__(self):
        super().__init__()

    def __call__(self, pred, gt):
        assert pred.dim() == gt.dim(), \
            "inconsistent dimensions, pred shape is {}, but gt shape is {}.".format(pred.shape, gt.shape)

        if pred.shape != gt.shape:
            pred = interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=True)

        valid_mask = (gt != 0).detach()
        diff = gt - pred
        diff = diff[valid_mask]
        self.loss = torch.pow(diff, 2).sum()
        return self.loss
