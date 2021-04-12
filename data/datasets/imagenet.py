import glob
import json
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import tqdm
import concurrent.futures
import pickle
import torch
from torchvision import transforms
from PIL import ImageFile

from data.data_processing.augment import augment_image

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImagenetDataset(Dataset):
    def __init__(self, config, mode):
        """
        A data provider class inheriting from Pytorch's Dataset class. It takes care of creating task sets for
        our few-shot learning model training and evaluation

        :param config: Arguments in the form of a ConfigNameSpace object. Includes all hyperparameters necessary for the
        data-provider.
        :param mode: mode string of dataset`s current domain: one of [source_train, source_val, target_train, target_val]
        """
        self.config = config
        self.domain, self.mode = mode.split('_')
        self.data_loaded_in_memory = False
        self.index = 0
        self.augment_images = False

        self.data_labels, self.data_paths, self.data_by_class_paths = self.load_dataset()

        self.indexes = 0
        self.dataset_size_dict = {key: len(self.data_by_class_paths[key]) for key in
                                  list(self.data_by_class_paths.keys())}
        self.label_set = self.get_label_set()
        self.data_length = np.sum([len(self.data_by_class_paths[key]) for key in self.data_by_class_paths.keys()])

        print(f'Datapoinst in {self.domain}/{self.mode}:', self.data_length)
        self.observed_seed_set = None
        self.update_seed()

    def set_random_state(self):
        train_rng = np.random.RandomState(seed=self.config.train_seed)
        train_seed = train_rng.randint(1, 999999)
        self.config.train_seed = train_seed
        self.init_seed = self.config.train_seed
        self.seed = self.config.train_seed

    def load_dataset(self):
        """
        Loads a dataset's dictionary files.
        in the config object.
        :return: The current dataset
        """
        data_labels, data_paths, data_by_class_paths, self.index_to_label_name_dict_file, self.label_to_index = \
            self.load_datapaths()

        if self.config.load_into_memory:
            print(f"Loading {self.mode} data into RAM")

            x_loaded = {key: np.zeros(len(value), ) for key, value in data_by_class_paths.items()}
            with tqdm.tqdm(total=len(data_by_class_paths)) as pbar_memory_load:
                with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
                    # Process the list of files, but split the work across the process pool to use all CPUs!
                    for (class_label, class_images_loaded) in executor.map(self.load_parallel_batch,
                                                                           (data_by_class_paths.items())):
                        x_loaded[class_label] = class_images_loaded
                        pbar_memory_load.update(1)

            x_labels = []
            x_paths = []
            for key, values in x_loaded.items():
                x_labels.extend([self.get_label_from_index(key)] * len(values))
                x_paths.extend(values)

            data_labels = x_labels
            data_paths = x_paths
            data_by_class_paths = x_loaded
            self.data_loaded_in_memory = True

        return data_labels, data_paths, data_by_class_paths

    def load_datapaths(self):
        """
        If saved json dictionaries of the data are available, then this method loads the dictionaries such that the
        data is ready to be read. If the json dictionaries do not exist, then this method calls get_data_paths()
        which will build the json dictionary containing the class to filepath samples, and then store them.
        :return: data_image_paths: dict containing class to filepath list pairs.
                 index_to_label_name_dict_file: dict containing numerical indexes mapped to the human understandable
                 string-names of the class
                 label_to_index: dictionary containing human understandable string mapped to numerical indexes
        """
        dataset_files_dir = 'dataset_files/imagenet84'
        data_labels_file = f"{dataset_files_dir}/{self.domain}/{self.mode}_imagenet84_labels.json"
        data_path_file = f"{dataset_files_dir}/{self.domain}/{self.mode}_imagenet84.json"
        data_by_class_path_file = f"{dataset_files_dir}/{self.domain}/{self.mode}_imagenet84_by_class.json"
        self.index_to_label_name_dict_file = f"{dataset_files_dir}/{self.domain}/" \
                                             f"{self.mode}_map_to_label_name_imagenet84.json"
        self.label_name_to_map_dict_file = f"{dataset_files_dir}/{self.domain}/" \
                                           f"{self.mode}_label_name_to_map_imagenet84.json"
        self.label_to_word = self.load_from_json(f"dataset_files/imagenet/label_to_word.json")

        try:
            data_labels = self.load_from_json(filename=data_labels_file)
            data_paths = self.load_from_json(filename=data_path_file)
            data_by_class_paths = self.load_from_json(filename=data_by_class_path_file)
            label_to_index = self.load_from_json(filename=self.label_name_to_map_dict_file)
            index_to_label_name_dict_file = self.load_from_json(filename=self.index_to_label_name_dict_file)
            return data_labels, data_paths, data_by_class_paths, index_to_label_name_dict_file, label_to_index
        except:
            print("Mapped data paths can't be found, remapping paths.")
            data_by_class_paths, code_to_label_name, label_name_to_code, data_paths, data_labels = self.get_data_paths()
            self.save_to_json(dict_to_store=data_labels, filename=data_labels_file)
            self.save_to_json(dict_to_store=data_paths, filename=data_path_file)
            self.save_to_json(dict_to_store=data_by_class_paths, filename=data_by_class_path_file)
            self.save_to_json(dict_to_store=code_to_label_name, filename=self.index_to_label_name_dict_file)
            self.save_to_json(dict_to_store=label_name_to_code, filename=self.label_name_to_map_dict_file)
            return self.load_datapaths()

    def save_to_json(self, filename, dict_to_store):
        parent_dir = os.path.join(*filename.split('/')[:-1])
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        with open(os.path.abspath(filename), 'w') as f:
            json.dump(dict_to_store, fp=f)

    def load_from_json(self, filename):
        with open(filename, mode="r") as f:
            load_dict = json.load(fp=f)

        return load_dict

    def get_data_paths(self):
        """
        Method that scans the dataset directory and generates class to image-filepath list dictionaries.
        :return: data_image_paths: dict containing class to filepath list pairs.
                 index_to_label_name_dict_file: dict containing numerical indexes mapped to the human understandable
                 string-names of the class
                 label_to_index: dictionary containing human understandable string mapped to numerical indexes
        """
        path_template = os.path.join(self.config.dataset_path, self.domain, '*', self.mode, '*.JPEG')
        print("Get images in a form: ", path_template)
        paths = sorted(glob.glob(path_template))
        data_paths = [os.path.abspath(path) for path in paths]
        labels = sorted({self.get_label_from_path(path) for path in data_paths})

        idx_to_label_name = {idx: label for idx, label in enumerate(labels)}
        label_name_to_idx = {label: idx for idx, label in enumerate(labels)}
        data_image_path_dict = {idx: [] for idx in list(idx_to_label_name.keys())}
        data_labels = []
        with tqdm.tqdm(total=len(data_paths)) as pbar_error:
            # Process the list of files and put them into a dictionary of classes with the list of the image paths
            for image_file in data_paths:
                pbar_error.update(1)
                label = self.get_label_from_path(image_file)
                data_labels.append(label)
                data_image_path_dict[label_name_to_idx[label]].append(image_file)

        return data_image_path_dict, idx_to_label_name, label_name_to_idx, data_paths, data_labels

    def get_label_set(self):
        """
        Generates a set containing all class numerical indexes
        :return: A set containing all class numerical indexes
        """
        # index_to_label_name_dict_file = self.load_from_json(filename=self.index_to_label_name_dict_file)
        return set(list(self.index_to_label_name_dict_file.keys()))

    def get_index_from_label(self, label):
        """
        Given a class's (human understandable) string, returns the numerical index of that class
        :param label: A string of a human understandable class contained in the dataset
        :return: An int containing the numerical index of the given class-string
        """
        # label_to_index = self.load_from_json(filename=self.label_name_to_map_dict_file)
        return self.label_to_index[label]

    def get_label_from_index(self, index):
        """
        Given an index return the human understandable label mapping to it.
        :param index: A numerical index (int)
        :return: A human understandable label (str)
        """
        # index_to_label_name = self.load_from_json(filename=self.index_to_label_name_dict_file)
        return self.index_to_label_name_dict_file[index]

    def get_label_from_path(self, filepath):
        """
        Given a path of an image generate the human understandable label for that image.
        :param filepath: The image's filepath
        :return: A human understandable label.
        """
        label_bits = filepath.split("/")
        label = "/".join([label_bits[idx] for idx in self.config.indexes_of_folders_indicating_class])
        if self.config.labels_as_int:
            label = int(label)
        return label

    def get_word_form_label(self, label):
        """
        Given a label the human understandable word for that image.
        :param label: label, like: n00015388
        :return: A human understandable word, like: `animal`
        """
        return self.label_to_word[label.split('/')[0]]

    def load_image(self, image_path):
        """
        Given an image filepath and the number of channels to keep, load an image and keep the specified channels
        :param image_path: The image's filepath
        :return: An image array of shape (h, w, channels), whose values range between 0.0 and 1.0.
        """
        if not self.data_loaded_in_memory:
            image = Image.open(image_path)
            image = image.resize(self.config.input_size).convert('RGB')
            image = np.array(image, np.float32)
            image = image / 255.0
        else:
            image = image_path

        return image

    def load_parallel_batch(self, inputs):
        """
        Load a batch of images, given a list of filepaths
        :return: A numpy array of images of shape batch, height, width, channels
        """
        class_label, batch_image_paths = inputs
        image_batch = []

        if self.data_loaded_in_memory:
            for image_path in batch_image_paths:
                image_batch.append(image_path)
            image_batch = np.array(image_batch, dtype=np.float32)
        else:
            image_batch = [self.load_image(image_path=image_path) for image_path in batch_image_paths]
            image_batch = np.array(image_batch, dtype=np.float32)
            image_batch = self.preprocess_data(image_batch)

        return class_label, image_batch

    def preprocess_data(self, x):
        """
        Preprocesses data such that their shapes match the specified structures
        :param x: A data batch to preprocess
        :return: A preprocessed data batch
        """
        x_shape = x.shape
        x = np.reshape(x, (-1, x_shape[-3], x_shape[-2], x_shape[-1]))
        if self.config.reverse_channels is True:
            reverse_photos = np.ones(shape=x.shape)
            for channel in range(x.shape[-1]):
                reverse_photos[:, :, :, x.shape[-1] - 1 - channel] = x[:, :, :, channel]
            x = reverse_photos
        x = x.reshape(x_shape)
        x = np.transpose(x, [0, 3, 1, 2])
        return x

    @staticmethod
    def reconstruct_original(x):
        """
        Applies the reverse operations that preprocess_data() applies such that the data returns to their original form
        :param x: A batch of data to reconstruct
        :return: A reconstructed batch of data
        """
        x = x * 255.0
        return x

    def get_set(self, seed, augment_images=False):
        """
        Generates a task-set to be used for training or evaluation
        :param set_name: The name of the set to use, e.g. "train", "val" etc.
        :return: A task-set containing an image and label support set, and an image and label target set.
        """
        # seed = seed % self.config.total_unique_tasks
        rng = np.random.RandomState(seed)
        selected_classes = rng.choice(list(self.dataset_size_dict.keys()),
                                      size=self.config.num_samples_per_class, replace=False)
        rng.shuffle(selected_classes)
        k_list = rng.randint(0, 4, size=self.config.num_samples_per_class)
        k_dict = {selected_class: k_item for (selected_class, k_item) in zip(selected_classes, k_list)}
        episode_labels = [i for i in range(self.config.num_samples_per_class)]
        class_to_episode_label = {selected_class: episode_label for (selected_class, episode_label) in
                                  zip(selected_classes, episode_labels)}

        x_images = []
        y_labels = []

        for class_entry in selected_classes:
            choose_samples_list = rng.choice(self.dataset_size_dict[class_entry],
                                             size=self.config.num_samples_per_class + self.config.num_target_samples,
                                             replace=False)
            class_image_samples = []
            class_labels = []
            for sample in choose_samples_list:
                choose_samples = self.data_by_class_paths[class_entry][sample]
                _, x_class_data = self.load_parallel_batch([None, choose_samples])[0]
                k = k_dict[class_entry]
                x_class_data = augment_image(image=x_class_data, k=k,
                                             channels=self.image_channel, augment_bool=augment_images,
                                             dataset_name=self.dataset_name, config=self.config)
                class_image_samples.append(x_class_data)
                class_labels.append(int(class_to_episode_label[class_entry]))
            class_image_samples = torch.stack(class_image_samples)
            x_images.append(class_image_samples)
            y_labels.append(class_labels)

        x_images = torch.stack(x_images)
        y_labels = np.array(y_labels, dtype=np.float32)

        support_set_images = x_images[:, :self.config.num_samples_per_class]
        support_set_labels = y_labels[:, :self.config.num_samples_per_class]
        target_set_images = x_images[:, self.config.num_samples_per_class:]
        target_set_labels = y_labels[:, self.config.num_samples_per_class:]

        return support_set_images, target_set_images, support_set_labels, target_set_labels, seed

    def __len__(self):
        if self.config.small:
            return 1000
        elif 'val' in self.mode:
            return self.data_length
        elif 'train' in self.mode:
            return self.config.iteration_number or self.data_length
        else:
            raise ValueError(f'Wrong mode: {self.mode}')

    def set_augmentation(self, augment_images):
        self.augment_images = augment_images

    def switch_set(self, current_iter=None):
        if self.mode == "train":
            self.update_seed(seed=self.init_seed + current_iter)

    def update_seed(self, seed=0):
        self.seed = seed

    def __getitem__(self, idx):
        if self.config.learning_type in ['transfer_learning', 'simple_learning', 'multitask_learning']:
            return self.get_transfer_dict(idx)

        elif self.config.learning_type == 'meta_learning':
            support_set_images, target_set_image, support_set_labels, target_set_label, seed = \
                self.get_set(seed=self.seed + idx, augment_images=self.augment_images)
            return {'support_set_images': support_set_images, 'target_set_image': target_set_image,
                    'support_set_labels': support_set_labels, 'target_set_label': target_set_label}

        else:
            raise ValueError(f'Wrong learning type: {self.config.learning_type}')

    def get_transfer_dict(self, idx):
        """
        Loading the data according to the transfer learning agenda

        :param idx: index of the data

        :return: dictionary containing the image and the target
        """
        path = self.data_paths[idx]
        label = self.data_labels[idx]
        index = self.get_index_from_label(label)
        index, x_class_data = self.load_parallel_batch([index, [path]])
        word = self.get_word_form_label(label)

        return {'image': x_class_data[0], 'target': index, 'word': word}
