import concurrent.futures
import glob
import json
import os
import sys
from PIL import Image
import numpy as np
import tqdm

from config import ConfigNamespace

# set current workdir to project dir
current_working_dir = os.getcwd()
new_working_dir = '/' + os.path.join(*current_working_dir.split('/')[:-2])
os.chdir(new_working_dir)
sys.path.insert(0, new_working_dir)


class ImagenetConverter:
    def __init__(self, config, orig_path, goal_path):
        self.orig_path = orig_path
        self.goal_path = goal_path
        self.config = config

        with open('dataset_files/imagenet/ilsvrc_2012_dataset_spec.json', 'r') as f:
            dataset_spec = json.load(f)

        train_classes = list(dataset_spec['images_per_class']['TRAIN'].keys())
        val_classes = list(dataset_spec['images_per_class']['VALID'].keys())
        # test_classes = list(dataset_spec['images_per_class']['TEST'].keys())

        classes = glob.glob(orig_path + '/train/*')
        classes = [class_path.split('/')[-1] for class_path in classes]

        self.classes_dict = dict()
        self.classes_dict['source'] = [train_class for train_class in train_classes if train_class in classes]
        self.classes_dict['target'] = [val_class for val_class in val_classes if val_class in classes]

        self.class_binding = dict()
        with open('dataset_files/imagenet/train_orig_goal_class_pairs.txt', 'r') as f:
            self.class_binding['source'] = {element.strip('\n').split()[0]: element.strip('\n').split()[1]
                                            for element in f.readlines()}
        with open('dataset_files/imagenet/val_orig_goal_class_pairs.txt', 'r') as f:
            self.class_binding['target'] = {element.strip('\n').split()[0]: element.strip('\n').split()[1]
                                            for element in f.readlines()}

    def load_and_process_image(self, image_path):
        image = Image.open(image_path)
        image = image.resize(self.config.input_size).convert('RGB')

        return image

    def save_image(self, image, image_dir, image_name):
        image_path = os.path.join(image_dir, image_name)
        image.save(image_path)

    def process_image(self, image_path, new_image_dir):
        image = self.load_and_process_image(image_path)
        image_name = image_path.split('/')[-1]
        self.save_image(image, new_image_dir, image_name)

    def convert(self):
        with open(os.path.join(self.orig_path, 'synset_labels.txt'), 'r') as f:
            val_labels = f.readlines()
        val_labels = [val_label.rstrip('\n') for val_label in val_labels]
        val_elements = glob.glob(os.path.join(self.orig_path, 'val', '*'))
        val_elements.sort()
        val_paths = {class_name: [] for class_name in np.unique(val_labels)}
        for val_label, val_element in zip(val_labels, val_elements):
            val_paths[val_label].append(val_element)

        for set_name in ['source', 'target']:
            print(f'Converting {set_name} set.')
            with tqdm.tqdm(total=len(self.classes_dict[set_name])) as pbar_memory_load:
                for class_name in self.classes_dict[set_name]:
                    train_elements = glob.glob(os.path.join(self.orig_path, 'train', class_name, '*'))
                    val_elements = val_paths[class_name]

                    # train
                    image_save_dir = os.path.join(self.goal_path, set_name,
                                                  self.class_binding[set_name][class_name], 'train')
                    if not os.path.exists(image_save_dir):
                        os.makedirs(image_save_dir)

                    if set_name == 'target':
                        indices = np.random.choice(np.arange(len(train_elements)),
                                                   self.config.target_train_number,
                                                   replace=False)
                        train_data_args = [(train_elements[i], image_save_dir) for i in indices]
                    else:
                        train_data_args = [(train_element, image_save_dir) for train_element in train_elements]
                    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
                        for _ in executor.map(self.process_image, *zip(*train_data_args)):
                            pass

                    # val
                    image_save_dir = os.path.join(self.goal_path, set_name,
                                                  self.class_binding[set_name][class_name], 'val')
                    if not os.path.exists(image_save_dir):
                        os.makedirs(image_save_dir)
                    for val_element in val_elements:
                        self.process_image(val_element, image_save_dir)

                    pbar_memory_load.update(1)


if __name__ == '__main__':
    config = ConfigNamespace({'input_size': [84, 84], 'target_train_number': 100})
    orig_path = '../data/imagenet'
    goal_path = '../data/imagenet84'
    imagenet_converter = ImagenetConverter(config, orig_path, goal_path)
    imagenet_converter.convert()
