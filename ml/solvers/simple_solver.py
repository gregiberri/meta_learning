#!/usr/bin/python3
# -*- coding: utf-8 -*-
import inspect
import logging
import torch
from ray import tune

from ml.models.full_model import BaseModel
from ml.optimizers import get_optimizer, get_lr_policy
from ml.solvers.base_solver import Solver
from data.datasets import get_dataloader
from ml.metrics.metrics import Metrics

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

RANDOM_SEED = 5


class SimpleSolver(Solver):
    def init_epochs(self):
        self.target_epoch = 0

        self.target_epochs = self.config.env.epochs

    def init_models(self):
        self.target_model = BaseModel(self.config.model)
        self.target_model.cuda()
        self.model_input_keys = [p.name for p in inspect.signature(self.target_model.forward).parameters.values()]

    def init_optimizers(self):
        self.target_optimizer = get_optimizer(self.config.optimizer, self.target_model.parameters())

    def init_lr_policies(self):
        self.target_lr_policy = get_lr_policy(self.config.lr_policy, optimizer=self.target_optimizer)

    def init_dataloaders(self):
        # get dataloaders
        self.target_train_loader = get_dataloader(self.config.target_data, 'target_train')
        self.target_val_loader = get_dataloader(self.config.target_data, 'target_val')

    def init_metrics(self):
        self.target_train_metric = Metrics(self.result_dir, tag='target_train', niter=len(self.target_train_loader))
        self.target_val_metric = Metrics(self.result_dir, tag='target_val', niter=len(self.target_val_loader))

    def load_checkpoint(self):
        if self.args.resume:
            self.current_mode = 'target_train'
            self.init_from_checkpoint()

    def get_model_string(self):
        return 'model'

    def __getattribute__(self, item):
        if item in ['loader', 'metric']:
            item = self.current_mode + '_' + item
        elif item in ['epoch', 'epochs', 'model', 'optimizer', 'lr_policy']:
            item = self.current_mode.split('_')[0] + '_' + item

        return super(Solver, self).__getattribute__(item)

    def __setattr__(self, key, value):
        if key in ['loader', 'metric']:
            key = self.current_mode + '_' + key
        elif key in ['epoch', 'epochs', 'model', 'optimizer', 'lr_policy']:
            key = self.current_mode.split('_')[0] + '_' + key

        return super(Solver, self).__setattr__(key, value)

    def run(self):
        if self.phase == 'train':
            self.current_mode = 'target_train'
            self.train()
        elif self.phase == 'val':
            self.current_mode = 'target_val'
            self.eval()
        else:
            raise ValueError(f'Wrong phase: {self.phase}')

        return self.target_val_metric

    def train(self):
        start_mode = self.current_mode
        # start epoch
        for self.epoch in range(self.epoch, self.epochs):
            self.current_mode = start_mode
            self.run_epoch()

            self.eval()
            self.save_best_checkpoint()

        # self.writer.close()

    def eval(self):
        self.current_mode = 'target_val'

        with torch.no_grad():
            self.run_epoch()
