#!/usr/bin/python3
# -*- coding: utf-8 -*-
import logging
import os
import sys

from ray import tune
from ray.tune.suggest.bayesopt import BayesOptSearch

from config import ConfigNamespace
from ml.solvers import get_solver

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

RANDOM_SEED = 5


class HPOSolver(object):

    def __init__(self, config, args):
        """
        Solver parent function to control the experiments.

        :param config: config namespace containing the experiment configuration
        :param args: arguments of the training
        """
        self.args = args
        self.phase = args.mode
        self.config = config

        self.experiment_number = 0

        self.init_results_dir()

    def init_results_dir(self):
        self.result_dir = os.path.join(self.config.env.result_dir, self.config.id)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

    def run(self):
        config_dict = self.config.dict()
        ray_result_dir = os.path.join(self.result_dir, 'tune')
        result = tune.run(self.run_experiment, config=config_dict, num_samples=100, checkpoint_at_end=True,
                          queue_trials=True, local_dir=ray_result_dir,
                          resources_per_trial={"cpu": 4, "gpu": 0.15})

        self.report_result(result)

    def report_result(self, result):
        best_trial = result.get_best_trial('accuracy', 'max', 'last')
        print("Best trial config: {}".format(best_trial.logdir))
        print("Best trial final validation accuracy: {}".format(best_trial.last_result['_metric']))

        result_file = os.path.join(self.result_dir, 'result.txt')
        with open(result_file, 'w') as f:
            f.write("Best trial logdir: \n{}".format(best_trial.logdir))
            f.write("\n\nBest trial final validation accuracy: \n{}".format(best_trial.last_result['_metric']))

    def run_experiment(self, search_space):
        # set current workdir to project dir (w/o it the datareader cant reach dataset_files)
        current_working_dir = __file__
        new_working_dir = '/' + os.path.join(*current_working_dir.split('/')[:-3])
        os.chdir(new_working_dir)
        sys.path.insert(0, new_working_dir)

        args = self.args
        args.mode = 'train'
        args.id_tag = tune.get_trial_dir().split('/')[-2:-1]

        config = ConfigNamespace(search_space)
        config.id = os.path.join(config.id, 'outputs')

        solver = get_solver(config, args)
        metric = solver.run()
        tune.report(loss=min(metric.epoch_results['loss']), accuracy=max(metric.epoch_results['acc']))

        return max(metric.epoch_results['acc'])
