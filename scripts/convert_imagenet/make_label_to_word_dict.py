# set current workdir to project dir
import csv
import glob
import json
import os
import sys
import numpy as np
import tqdm

current_working_dir = os.getcwd()
new_working_dir = '/' + os.path.join(*current_working_dir.split('/')[:-2])
os.chdir(new_working_dir)
sys.path.insert(0, new_working_dir)

np.random.seed(0)


class MakeLabelToWordDict:
    def __init__(self):
        with open('scripts/convert_imagenet/dataset_specs/ilsvrc_2012_dataset_spec.json', 'r') as f:
            dataset_spec = json.load(f)

        self.train_nodes = dataset_spec['split_subgraphs']['TRAIN']
        self.val_nodes = dataset_spec['split_subgraphs']['VALID']
        self.test_nodes = dataset_spec['split_subgraphs']['TEST']

        self.goal_dir = 'dataset_files/imagenet/'
        if not os.path.exists(self.goal_dir):
            os.makedirs(self.goal_dir)

    def make(self):
        labeldict = dict()

        for element in self.train_nodes:
            labeldict[element['wn_id']] = element['words'].split(',')[0]
        for element in self.val_nodes:
            labeldict[element['wn_id']] = element['words'].split(',')[0]
        for element in self.test_nodes:
            labeldict[element['wn_id']] = element['words'].split(',')[0]

        file_path = os.path.join(self.goal_dir, 'label_to_word.json')
        with open(file_path, 'w') as f:
            json.dump(labeldict, f)


if __name__ == '__main__':
    make_label_to_word_dict = MakeLabelToWordDict()
    make_label_to_word_dict.make()
