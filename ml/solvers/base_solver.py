#!/usr/bin/python3
# -*- coding: utf-8 -*-
import gc
import logging
import sys
import os
import time
import numpy as np
import random
import torch

from ml.utils.pyt_io import load_model
from ml.utils.pyt_ops import tensor2cuda
from tqdm import tqdm

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

RANDOM_SEED = 5


class Solver(object):

    def __init__(self, config, args):
        """
        Solver parent function to control the experiments.

        :param config: config namespace containing the experiment configuration
        :param args: arguments of the training
        """
        self.args = args
        self.phase = args.mode
        self.config = config

        self.init_epochs()

        self._set_seed()

        self.init_results_dir()
        self.init_dataloaders()
        self.init_models()
        self.init_optimizers()
        self.init_lr_policies()
        self.init_metrics()

        self.load_checkpoint()

        # self.visualizer = Visualizer(self.writer)

    def init_results_dir(self):
        result_name = os.path.join(self.config.id, self.args.id_tag) if self.args.id_tag else self.config.id
        self.result_dir = os.path.join(self.config.env.result_dir, result_name)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

    @property
    def learning_rates(self):
        lrs = []
        for i in range(len(self.optimizer.param_groups)):
            lrs.append(self.optimizer.param_groups[i]['lr'])
        return lrs

    def init_epochs(self):
        """
        This function should implement the epoch number initialization(s).
        """
        raise NotImplementedError()

    def init_models(self):
        """
        This function should implement the model initialization(s).
        """
        raise NotImplementedError()

    def init_optimizers(self):
        """
        This function should implement the optimizer initialization(s).
        """
        raise NotImplementedError()

    def init_lr_policies(self):
        """
        This function should implement the learnin rate policy initialization(s).
        """
        raise NotImplementedError()

    def init_dataloaders(self):
        """
        This function should implement the dataloader initialization(s).
        """
        raise NotImplementedError()

    def init_metrics(self):
        """
        This function should implement the metric initialization(s).
        """
        raise NotImplementedError()

    def load_checkpoint(self):
        """
        This function should implement the checkpoint loader initialization(s).
        """
        raise NotImplementedError()

    def get_model_string(self):
        """
        This function should implement the string containing the current model`s name initialization(s).

        :return model_str: string containing the current model`s name
        """
        raise NotImplementedError()

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

    def before_epoch(self):
        self.model.set_domain(self.current_mode)
        if 'train' in self.current_mode:
            self.model.train()
        elif 'val' in self.current_mode:
            self.model.eval()
        else:
            raise ValueError(f'Wrong solver mode: {self.current_mode}')

        torch.cuda.empty_cache()

    def after_epoch(self):
        gc.collect()
        self.metric.on_epoch_end(self.epoch)
        torch.cuda.empty_cache()

    def run_epoch(self):
        """
        Run a full epoch according to the current self.current_mode
        """
        self.before_epoch()

        # set loading bar
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        with tqdm(range(len(self.loader)), file=sys.stdout, bar_format=bar_format, position=0, leave=True) as pbar:
            # start measuring preproc time
            t_start = time.time()
            for idx, minibatch in enumerate(self.loader):
                model_input_dict = self.get_model_input_dict(minibatch)
                preproc_time = time.time() - t_start

                # start measuring the train time
                t_start = time.time()
                # train
                output_dict = self.step(**model_input_dict)
                cmp_time = time.time() - t_start

                self.metric.compute_metric(output_dict)

                print_str = f'[{self.current_mode}] Epoch {self.epoch}/{self.epochs} ' \
                            + f'Iter{idx + 1}/{len(self.loader)}: ' \
                            + f'lr={self.learning_rates[0]:.8f} ' \
                            + self.metric.get_snapshot_info() \
                            + f'prepr: {preproc_time:.3f}s ' \
                            + f'infer: {cmp_time:.3f}s '
                pbar.set_description(print_str, refresh=False)
                pbar.update(1)

                # write on tensorboard
                # if idx % self.config.env.save_train_frequency == 0:
                #     self.visualizer.visualize(minibatch, pred, self.epoch, tag='train')
                #     metric.add_scalar(self.writer, iteration=idx)

                # start measuring preproc time
                t_start = time.time()

        self.after_epoch()

    def step(self, **model_inputs):
        """
        Make one iteration step: either a train or a val step

        :param model_inputs:
        :return: dictionary of prediction and loss
        """
        output_dict = self.model(**model_inputs)

        if 'train' in self.current_mode:
            # backward
            self.optimizer.zero_grad()
            output_dict['loss'].backward()
            self.optimizer.step()
            self.lr_policy.step(self.epoch)

        return output_dict

    def init_from_checkpoint(self):
        path = os.path.join(self.result_dir, self.get_model_string() + '.pth')
        continue_state_object = torch.load(path, map_location=torch.device("cpu"))

        self.config = continue_state_object['config']
        self.metric = continue_state_object['metric']
        self.epoch = continue_state_object['epoch']

        load_model(self.model, continue_state_object['model'], distributed=False)
        self.model.cuda()

        self.optimizer.load_state_dict(continue_state_object['optimizer'])
        self.lr_policy.load_state_dict(continue_state_object['lr_policy'])

        del continue_state_object
        logging.info("Init trainer from checkpoint")

    def save_best_checkpoint(self):
        """
        Save the model if the last epoch result is the best.
        """
        epoch_results = self.metric.epoch_results

        if not min(epoch_results['acc']) == epoch_results['acc'][-1]:
            return

        path = os.path.join(self.result_dir, self.get_model_string() + '.pth')

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
        state_dict['metric'] = self.metric

        torch.save(state_dict, path)
        del state_dict
        del new_state_dict
        logging.info(f"\nSave checkpoint to file {path}")
