#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2019-12-24 21:47
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : solver.py
"""
import gc
import logging
import sys
import os
import time
import numpy as np
import random
import torch

from ml.models import get_model
from ml.utils.pyt_ops import tensor2cuda
from tqdm import tqdm

from config import ConfigNamespace
from data.datasets import get_dataloader
from ml.metrics.metrics import Metrics

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

RANDOM_SEED = 5


class Solver(object):

    def __init__(self, args):
        """
            :param config: easydict
        """
        self.epoch = 0
        self.iteration = 0
        self.config = None
        self.result_dir = None
        self.model, self.optimizer, self.lr_policy = None, None, None
        self.writer = None
        self.model_input_keys = None
        self.best_rmses = []

        self.args = args

        self.config = ConfigNamespace(args.config)

        self.get_dataloaders()

        self.model = get_model(self.config.model)

        # self.loss_meter = AverageMeter()
        # self.train_metric = Metrics(self.result_dir, tag='train', niter=self.niter_train)
        # self.val_metric = Metrics(self.result_dir, tag='val', niter=self.niter_test)
        # self.visualizer = Visualizer(self.writer)

    def get_dataloaders(self):
        # get dataloaders
        self.train_loader, self.niter_train = get_dataloader(self.config.data, current_set='train')
        self.test_loader, self.niter_test = get_dataloader(self.config.data, current_set='val')

    def _set_seed(self):
        """
        Set the random seeds
        """
        torch.manual_seed(RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)

    def get_model_input_dict(self, minibatch):
        model_input_dict = {k: v for k, v in minibatch.items() if k in self.model_input_keys}
        if torch.cuda.is_available():
            model_input_dict = tensor2cuda(model_input_dict)
        else:
            raise SystemError('No cuda device found.')
        return model_input_dict

    def get_loader(self, mode, scenes=None):
        if mode == 'train':
            return self.train_loader, self.niter_train
        elif mode == 'val':
            return self.test_loader, self.niter_test
        elif mode == 'scene_retrain':
            scene_avg = np.median([val for key, val in scenes.items()])
            retrain_scenes = [key for key, val in scenes.items() if val > scene_avg]
            return get_dataloader(self.config, current_set=True, scenes=retrain_scenes)
        else:
            raise ValueError(f'Wrong solver mode: {mode}')

    def get_metric(self, mode, niter=None):
        if mode == 'train':
            return self.train_metric
        elif mode == 'val':
            return self.val_metric
        elif mode == 'scene_retrain':
            return Metrics(self.result_dir, tag='bad_scene', niter=niter)
        else:
            raise ValueError(f'Wrong solver mode: {mode}')

    def run(self):
        if self.args.mode == 'train':
            self.train()
        elif self.args.mode == 'val':
            self.eval()
        else:
            raise ValueError()

    def train(self):
        # start epoch
        for self.epoch in range(self.epoch, self.config.env.epochs):
            self.before_epoch()
            self.run_epoch(mode='train')
            self.after_epoch()

            # self.train_metric.on_epoch_end()
            # validation
            # first set the val dataset lidar sparsity to the train data current one
            # self.val_loader.dataset.lidar_sparsity = self.train_loader.dataset.lidar_sparsity
            self.eval()
            self.save_best_checkpoint(self.val_metric.epoch_results)

        self.writer.close()

        return min(self.train_metric.epoch_results['rmse'])  # best value

    @torch.no_grad()
    def eval(self):
        self.run_epoch(mode='val')
        logging.info(f'After Epoch {self.epoch}/{self.config.env.epochs}, {self.val_metric.get_result_info()}')
        self.val_metric.on_epoch_end()

    def run_epoch(self, mode='train', scenes=None):
        loader, niter = self.get_loader(mode, scenes=scenes)
        metric = self.get_metric(mode, niter=niter)
        epoch_iterator = iter(loader)

        # set loading bar
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(niter), file=sys.stdout, bar_format=bar_format)
        for idx in pbar:
            # start measuring preproc time
            t_start = time.time()

            # get the minibatch and filter out to the input and gt elements
            minibatch = epoch_iterator.next()
            model_input_dict = self.get_model_input_dict(minibatch)

            # preproc time
            t_end = time.time()
            preproc_time = t_end - t_start

            # start measuring the train time
            t_start = time.time()

            # train
            pred, loss = self.step(mode, **model_input_dict)

            # train time
            t_end = time.time()
            cmp_time = t_end - t_start
            if loss is not None and mode == 'train':
                self.loss_meter.update(loss)
                self.writer.add_scalar("train/loss", self.loss_meter.mean(), self.epoch)
            else:
                loss = torch.as_tensor(0)
            metric.compute_metric(pred, model_input_dict, minibatch['scene'])

            print_str = f'[{mode}] Epoch {self.epoch}/{self.config.env.epochs} ' \
                        + f'Iter{idx + 1}/{niter}: ' \
                        + f'lr={self.get_learning_rates()[0]:.8f} ' \
                        + f'losses={loss.item():.2f}({self.loss_meter.mean():.2f}) ' \
                        + metric.get_snapshot_info() \
                        + f' prp: {preproc_time:.2f}s ' \
                        + f'inf :{cmp_time:.2f}s '
            pbar.set_description(print_str, refresh=False)

            if idx % self.config.env.save_train_frequency == 0:
                self.visualizer.visualize(minibatch, pred, self.epoch, tag='train')
                metric.add_scalar(self.writer, iteration=idx)

    def step(self, mode='train', **model_inputs):
        """
        :param model_inputs:
        :return:
        """
        if mode == 'val':
            with torch.no_grad():
                pred = self.model(**model_inputs)

            return pred['pred'], None

        elif mode == 'train' or mode == 'scene_retrain':
            self.iteration += 1
            output_dict = self.model(**model_inputs)

            pred = output_dict['pred']
            loss = output_dict['loss']

            # backward
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.lr_policy.step(self.epoch)

            return pred, loss.data

    def before_epoch(self):
        self.iteration = 0
        self.epoch = self.epoch
        self.model.train()
        torch.cuda.empty_cache()

    def after_epoch(self):
        self.model.eval()
        gc.collect()
        torch.cuda.empty_cache()

    def save_best_checkpoint(self, epoch_results):
        if not min(epoch_results['irmse']) == epoch_results['irmse'][-1]:
            return

        path = os.path.join(self.result_dir, 'model_best.pth')

        t_start = time.time()

        state_dict = {}

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in self.model.state_dict().items():
            key = k
            if k.split('.')[0] == 'module':
                key = k[7:]
            new_state_dict[key] = v

        state_dict['config'] = self.config
        state_dict['model'] = new_state_dict
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['lr_policy'] = self.lr_policy.state_dict()
        state_dict['epoch'] = self.epoch
        state_dict['iteration'] = self.iteration

        t_iobegin = time.time()
        torch.save(state_dict, path)
        del state_dict
        del new_state_dict
        t_end = time.time()
        logging.info(
            "Save checkpoint to file {}, "
            "Time usage:\n\tprepare snapshot: {}, IO: {}".format(
                path, t_iobegin - t_start, t_end - t_iobegin))

    def get_learning_rates(self):
        lrs = []
        for i in range(len(self.optimizer.param_groups)):
            lrs.append(self.optimizer.param_groups[i]['lr'])
        return lrs
