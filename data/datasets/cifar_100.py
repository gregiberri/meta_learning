import logging
import os
import pickle

import numpy as np
import tqdm
from torch.utils.data import Dataset
from torchvision.transforms import transforms

MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


class CIFAR100Dataset(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, config, mode='train'):
        # if transform is given, we transoform data using
        self.config = config
        self.mode = mode
        self.data_loaded_in_memory = False

        self.datasets = self.load_dataset()

        self.index = 0
        self.dataset_size_dict = {key: len(self.datasets[key]) for key in list(self.datasets.keys())}
        # self.label_set = self.get_label_set()
        self.data_length = np.sum([len(self.datasets[key]) for key in self.datasets.keys()])

        print("data", self.data_length)
        self.observed_seed_set = None

        self.get_transform()

    def get_transform(self):
        if self.mode == 'train':
            self.transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.RandomRotation(15),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(MEAN, STD)])
        elif self.mode == 'val':
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(MEAN, STD)])
        else:
            raise ValueError(f'Wrong mode: {self.mode}')

    def set_random_state(self):
        train_rng = np.random.RandomState(seed=self.config.train_seed)
        train_seed = train_rng.randint(1, 999999)
        self.config.train_seed = train_seed
        self.init_seed = self.config.train_seed
        self.seed = self.config.train_seed

    def load_dataset(self):
        data_path = self.load_datapaths()

        with open(data_path, 'rb') as cifar100:
            all_data = pickle.load(cifar100, encoding='bytes')

        if self.config.load_into_memory:
            print(f"Loading {self.mode} data into RAM")
            data = all_data['data'.encode()]
            labels = all_data['fine_labels'.encode()]
            x_loaded = {label_idx: [] for label_idx in range(self.config.num_classes)}
            with tqdm.tqdm(total=len(data)) as pbar_memory_load:
                for data, label in zip(data, labels):
                    image = self.load_parallel_batch(data)
                    x_loaded[label].append(image)
                    pbar_memory_load.update(1)
            all_data = {key: np.array(value) for key, value in x_loaded.items()}
            self.data_loaded_in_memory = True
        else:
            raise RuntimeError('For CIFAR dataset the data should be always loaded to the memory (set load_into_memory '
                               'true in the config file.')

        return all_data

    def load_datapaths(self):
        if self.mode == 'train':
            return os.path.join(self.config.dataset_path, 'train')
        elif self.mode == 'val':
            return os.path.join(self.config.dataset_path, 'test')
        else:
            raise ValueError(f'Wrong mode: {self.mode}')

    def load_image(self, image):
        """
        Given the image data loaded from the pickle file loads the image
        :param image: The image data from the pickle file
        :return: An image array of shape (h, w, channels), whose values range between 0.0 and 1.0.
        """
        if not self.data_loaded_in_memory:
            r = image[:1024].reshape(32, 32)
            g = image[1024:2048].reshape(32, 32)
            b = image[2048:].reshape(32, 32)
            image = np.dstack((r, g, b))
            image = image / 255.0
        else:
            image = image
        return image

    def make_onehot(self, label):
        """
        Given the label index make the one-hot vector
        :param label: The image data from the pickle file
        :return: A label array of shape (N_class), whose values are 1.0. for the gt class and 0.0 otherwise
        """
        if not self.data_loaded_in_memory:
            one_hot_label = np.zeros(self.config.num_classes)
            one_hot_label[label] = 1.0
        else:
            one_hot_label = label
        return one_hot_label

    def load_parallel_batch(self, data):
        """
        Load a batch of images, given a list of unprocessed data and labels
        :return: A numpy array of images of shape batch, height, width, channels
        """

        if self.data_loaded_in_memory:
            image_batch = np.array(data, dtype=np.float32)
        else:
            image_batch = self.load_image(data)

            image_batch = np.array(image_batch, dtype=np.float32)

            image_batch = self.preprocess_data(image_batch)

        return image_batch

    def preprocess_data(self, x):
        """
        Preprocesses data such that their shapes match the specified structures
        :param x: A data batch to preprocess
        :return: A preprocessed data batch
        """
        x_shape = x.shape
        x = np.reshape(x, (-1, x_shape[-3], x_shape[-2], x_shape[-1]))
        if self.config.reverse_channels:
            reverse_photos = np.ones(shape=x.shape)
            for channel in range(x.shape[-1]):
                reverse_photos[:, :, :, x.shape[-1] - 1 - channel] = x[:, :, :, channel]
            x = reverse_photos
        x = x.reshape(x_shape)
        return x

    def reconstruct_original(self, x):
        """
        Applies the reverse operations that preprocess_data() applies such that the data returns to their original form
        :param x: A batch of data to reconstruct
        :return: A reconstructed batch of data
        """
        x = x * 255.0
        return x

    def __len__(self):
        return self.data_length // self.config.batch_size

    def __getitem__(self, index):
        label = self.data['fine_labels'.encode()][index]

        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        image = np.dstack((r, g, b))

        image = self.transform(image)

        return image, label
