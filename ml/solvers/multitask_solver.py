#!/usr/bin/python3
# -*- coding: utf-8 -*-
import inspect
import logging
import torch

from ml.models.full_model import BaseModel
from ml.optimizers import get_optimizer, get_lr_policy
from ml.solvers.base_solver import Solver
from data.datasets import get_dataloader
from ml.metrics.metrics import Metrics

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

RANDOM_SEED = 5


class MergedDataset:
    def __init__(self, source_train_loader, target_train_loader):
        self.len = min(len(source_train_loader), len(target_train_loader))

        self.source = iter(source_train_loader)
        self.target = iter(target_train_loader)

    def __getitem__(self, i):
        source_data = next(self.source)
        target_data = next(self.target)

        datas = {key: [source_data[key], target_data[key]] for key in source_data.keys()}
        datas['image'] = torch.cat(datas['image'], dim=0)

        return datas

    def __len__(self):
        return self.len


class MultitaskSolver(Solver):
    def init_epochs(self):
        self.epoch = 0
        self.epochs = self.config.env.epochs

    def init_models(self):
        self.config.model.params.data_split = [self.config.source_data.params.batch_size,
                                               self.config.target_data.params.batch_size]

        self.model = BaseModel(self.config.model)
        self.model.cuda()
        self.model_input_keys = [p.name for p in inspect.signature(self.model.forward).parameters.values()]

    def init_optimizers(self):
        self.optimizer = get_optimizer(self.config.multitask_optimizer, self.model.parameters())

    def init_lr_policies(self):
        self.lr_policy = get_lr_policy(self.config.multitask_lr_policy, optimizer=self.optimizer)

    def init_dataloaders(self):
        # get dataloaders
        self.source_train_loader = get_dataloader(self.config.source_data, 'source_train')
        self.source_val_loader = get_dataloader(self.config.source_data, 'source_val')
        self.target_train_loader = get_dataloader(self.config.target_data, 'target_train')
        self.target_val_loader = get_dataloader(self.config.target_data, 'target_val')

        self.multitask_train_loader = self.merge_dataloaders()

    def merge_dataloaders(self):
        dataloader = MergedDataset(self.source_train_loader, self.target_train_loader)

        return dataloader

    def init_metrics(self):
        self.multitask_train_metric = Metrics(self.result_dir, tag='multitask_train')
        self.source_val_metric = Metrics(self.result_dir, tag='source_val')
        self.target_val_metric = Metrics(self.result_dir, tag='target_val')

    def load_checkpoint(self):
        if self.args.resume:
            self.current_mode = 'multitask_train'
            self.init_from_checkpoint()

    def get_model_string(self):
        return 'model'

    def __getattribute__(self, item):
        if item in ['loader', 'metric']:
            item = self.current_mode + '_' + item

        return super(Solver, self).__getattribute__(item)

    def __setattr__(self, key, value):
        if key in ['loader', 'metric']:
            key = self.current_mode + '_' + key

        return super(Solver, self).__setattr__(key, value)

    def run(self):
        if self.phase == 'train':
            self.current_mode = 'multitask_train'
            self.train()
        elif self.phase == 'source_val':
            self.current_mode = 'source_val'
            self.eval()
        elif self.phase == 'target_val':
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
            self.multitask_train_loader = self.merge_dataloaders()
            self.run_epoch()

            self.eval()
            self.current_mode = start_mode
        # self.writer.close()

    def eval(self):
        self.current_mode = 'target_val'

        with torch.no_grad():
            self.run_epoch()
            self.save_best_checkpoint()
