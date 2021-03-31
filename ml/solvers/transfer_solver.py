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


class TransferSolver(Solver):
    def init_epochs(self):
        self.source_epoch = 0
        self.target_epoch = 0

        self.source_epochs = self.config.env.epochs
        self.target_epochs = self.config.env.target_epochs

    def init_models(self):
        self.source_model = BaseModel(self.config.model)
        self.source_model.cuda()
        self.model_input_keys = [p.name for p in inspect.signature(self.source_model.forward).parameters.values()]

        self.target_model = BaseModel(self.config.model)
        self.target_model.cuda()

    def init_optimizers(self):
        self.source_optimizer = get_optimizer(self.config.source_optimizer, self.source_model.parameters())
        self.target_optimizer = get_optimizer(self.config.target_optimizer, self.target_model.parameters())

    def init_lr_policies(self):
        self.source_lr_policy = get_lr_policy(self.config.source_lr_policy, optimizer=self.source_optimizer)
        self.target_lr_policy = get_lr_policy(self.config.target_lr_policy, optimizer=self.target_optimizer)

    def init_dataloaders(self):
        # get dataloaders
        self.source_train_loader = get_dataloader(self.config.source_data, 'source_train')
        self.source_val_loader = get_dataloader(self.config.source_data, 'source_val')
        self.target_train_loader = get_dataloader(self.config.target_data, 'target_train')
        self.target_val_loader = get_dataloader(self.config.target_data, 'target_val')

    def init_metrics(self):
        self.source_train_metric = Metrics(self.result_dir, tag='source_train', niter=len(self.source_train_loader))
        self.source_val_metric = Metrics(self.result_dir, tag='source_val', niter=len(self.source_val_loader))
        self.target_train_metric = Metrics(self.result_dir, tag='target_train', niter=len(self.target_train_loader))
        self.target_val_metric = Metrics(self.result_dir, tag='target_val', niter=len(self.target_val_loader))

    def load_checkpoint(self):
        if self.args.resume:
            self.current_mode = 'source_train'
            self.init_from_checkpoint()
            self.current_mode = 'target_train'
            self.init_from_checkpoint()

    def get_model_string(self):
        if 'source' in self.current_mode:
            return 'source_model'
        elif 'target' in self.current_mode:
            return 'target_model'
        else:
            raise ValueError(f'Wrong mode: {self.current_mode}')

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
            self.current_mode = 'source_train'
            self.train()
        elif self.phase == 'source_val':
            self.current_mode = 'source_val'
            self.eval()
        elif self.phase == 'adapt':
            self.current_mode = 'target_train'
            self.adapt()
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
            self.run_epoch()

            self.eval()

        self.writer.close()

    def eval(self):
        self.current_mode = self.current_mode.split('_')[0] + '_val'

        with torch.no_grad():
            self.run_epoch()
            self.save_best_checkpoint()

        if 'source' in self.current_mode:
            self.adapt()

    def adapt(self):
        self.current_mode = 'target_train'
        # reset the optimizer for every adaptation
        self.optimizer = get_optimizer(self.config.target_optimizer, self.model.parameters())
        self.lr_policy = get_lr_policy(self.config.target_lr_policy, optimizer=self.optimizer)

        # copy the model trained on the source domain
        self.model.load_state_dict(self.source_model.state_dict())
        for self.epoch in range(self.epochs):
            # adapt (train) the model on the target domain
            self.current_mode = 'target_train'
            self.run_epoch()

            # evaluate the model on the target domain
            self.current_mode = 'target_val'
            with torch.no_grad():
                self.run_epoch()
                self.save_best_checkpoint()
