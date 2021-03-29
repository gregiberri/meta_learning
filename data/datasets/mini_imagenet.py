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


class MiniImagenetDataset(Dataset):
    def __init__(self, config, mode):
        """
        A data provider class inheriting from Pytorch's Dataset class. It takes care of creating task sets for
        our few-shot learning model training and evaluation
        :param config: Arguments in the form of a Bunch object. Includes all hyperparameters necessary for the
        data-provider. For transparency and readability reasons to explicitly set as self.object_name all arguments
        required for the data provider, such that the reader knows exactly what is necessary for the data provider/
        """
        self.config = config
        self.mode = mode
        self.data_loaded_in_memory = False

        self.index = 0

        self.augment_images = False

        self.datasets = self.load_dataset()

        self.indexes = 0
        self.dataset_size_dict = {key: len(self.datasets[key]) for key in list(self.datasets.keys())}
        self.label_set = self.get_label_set()
        self.data_length = np.sum([len(self.datasets[key]) for key in self.datasets.keys()])

        print("data", self.data_length)
        self.observed_seed_set = None

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
        # rng = np.random.RandomState(seed=self.seed['val'])

        data_image_paths, index_to_label_name_dict_file, label_to_index = self.load_datapaths()

        if self.config.load_into_memory:
            print(f"Loading {self.mode} data into RAM")

            x_loaded = {key: np.zeros(len(value), ) for key, value in data_image_paths.items()}
            with tqdm.tqdm(total=len(data_image_paths)) as pbar_memory_load:
                with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
                    # Process the list of files, but split the work across the process pool to use all CPUs!
                    # self.config = SimpleNamespace(**self.config.dict())
                    for (class_label, class_images_loaded) in executor.map(self.load_parallel_batch,
                                                                           (data_image_paths.items())):
                        x_loaded[class_label] = class_images_loaded
                        pbar_memory_load.update(1)

            data_image_paths = x_loaded
            self.data_loaded_in_memory = True

        return data_image_paths

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
        dataset_dir = 'dataset_files/mini_imagenet'
        data_path_file = f"{dataset_dir}/{self.mode}_mini_imagenet_full_size.json"
        self.index_to_label_name_dict_file = f"{dataset_dir}/{self.mode}_map_to_label_name_mini_imagenet_full_size.json"
        self.label_name_to_map_dict_file = f"{dataset_dir}/{self.mode}_label_name_to_map_mini_imagenet_full_size.json"

        try:
            data_image_paths = self.load_from_json(filename=data_path_file)
            label_to_index = self.load_from_json(filename=self.label_name_to_map_dict_file)
            index_to_label_name_dict_file = self.load_from_json(filename=self.index_to_label_name_dict_file)
            return data_image_paths, index_to_label_name_dict_file, label_to_index
        except:
            print("Mapped data paths can't be found, remapping paths..")
            data_image_paths, code_to_label_name, label_name_to_code = self.get_data_paths()
            self.save_to_json(dict_to_store=data_image_paths, filename=data_path_file)
            self.save_to_json(dict_to_store=code_to_label_name, filename=self.index_to_label_name_dict_file)
            self.save_to_json(dict_to_store=label_name_to_code, filename=self.label_name_to_map_dict_file)
            return self.load_datapaths()

    def save_to_json(self, filename, dict_to_store):
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
        path_template = os.path.join(self.config.dataset_path, self.mode, '*', '*.jpg')
        print("Get images in a form: ", path_template)
        paths = sorted(glob.glob(path_template))
        data_image_path_list_raw = [os.path.abspath(path) for path in paths]
        labels = sorted({self.get_label_from_path(path) for path in data_image_path_list_raw})

        idx_to_label_name = {idx: label for idx, label in enumerate(labels)}
        label_name_to_idx = {label: idx for idx, label in enumerate(labels)}
        data_image_path_dict = {idx: [] for idx in list(idx_to_label_name.keys())}
        with tqdm.tqdm(total=len(data_image_path_list_raw)) as pbar_error:
            # Process the list of files and put them into a dictionary of classes with the list of the image paths
            for image_file in data_image_path_list_raw:
                pbar_error.update(1)
                label = self.get_label_from_path(image_file)
                data_image_path_dict[label_name_to_idx[label]].append(image_file)

        return data_image_path_dict, idx_to_label_name, label_name_to_idx

    def get_label_set(self):
        """
        Generates a set containing all class numerical indexes
        :return: A set containing all class numerical indexes
        """
        index_to_label_name_dict_file = self.load_from_json(filename=self.index_to_label_name_dict_file)
        return set(list(index_to_label_name_dict_file.keys()))

    def get_index_from_label(self, label):
        """
        Given a class's (human understandable) string, returns the numerical index of that class
        :param label: A string of a human understandable class contained in the dataset
        :return: An int containing the numerical index of the given class-string
        """
        label_to_index = self.load_from_json(filename=self.label_name_to_map_dict_file)
        return label_to_index[label]

    def get_label_from_index(self, index):
        """
        Given an index return the human understandable label mapping to it.
        :param index: A numerical index (int)
        :return: A human understandable label (str)
        """
        index_to_label_name = self.load_from_json(filename=self.index_to_label_name_dict_file)
        return index_to_label_name[index]

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
                image_batch.append(np.copy(image_path))
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
        return x

    def reconstruct_original(self, x):
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
                choose_samples = self.datasets[class_entry][sample]
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
        return self.data_length // self.config.batch_size

    def set_augmentation(self, augment_images):
        self.augment_images = augment_images

    def switch_set(self, current_iter=None):
        if self.mode == "train":
            self.update_seed(seed=self.init_seed + current_iter)

    def update_seed(self, seed=100):
        self.seed = seed

    def __getitem__(self, idx):
        support_set_images, target_set_image, support_set_labels, target_set_label, seed = \
            self.get_set(seed=self.seed + idx, augment_images=self.augment_images)

        return support_set_images, target_set_image, support_set_labels, target_set_label, seed

# class MetaLearningSystemDataLoader(object):
#     def __init__(self, config, current_iter=0):
#         """
#         Initializes a meta learning system dataloader. The data loader uses the Pytorch DataLoader class to parallelize
#         batch sampling and preprocessing.
#         :param config: An arguments NamedTuple containing all the required arguments.
#         :param current_iter: Current iter of experiment. Is used to make sure the data loader continues where it left
#         of previously.
#         """
#         self.num_of_gpus = config.num_of_gpus
#         self.batch_size = config.batch_size
#         self.samples_per_iter = config.samples_per_iter
#         self.num_workers = config.num_dataprovider_workers
#         self.total_train_iters_produced = 0
#         self.dataset = MiniImagenetDataset(config=config)
#         self.batches_per_iter = config.samples_per_iter
#         self.full_data_length = self.dataset.data_length
#         self.continue_from_iter(current_iter=current_iter)
#         self.config = config
#
#     def get_dataloader(self):
#         """
#         Returns a data loader with the correct set (train, val or test), continuing from the current iter.
#         :return:
#         """
#         return DataLoader(self.dataset, batch_size=(self.num_of_gpus * self.batch_size * self.samples_per_iter),
#                           shuffle=False, num_workers=self.num_workers, drop_last=True)
#
#     def continue_from_iter(self, current_iter):
#         """
#         Makes sure the data provider is aware of where we are in terms of training iterations in the experiment.
#         :param current_iter:
#         """
#         self.total_train_iters_produced += (current_iter * (self.num_of_gpus * self.batch_size * self.samples_per_iter))
#
#     def get_train_batches(self, total_batches=-1, augment_images=False):
#         """
#         Returns a training batches data_loader
#         :param total_batches: The number of batches we want the data loader to sample
#         :param augment_images: Whether we want the images to be augmented.
#         """
#         if total_batches == -1:
#             self.dataset.data_length = self.full_data_length
#         else:
#             self.dataset.data_length["train"] = total_batches * self.dataset.batch_size
#         self.dataset.switch_set(set_name="train", current_iter=self.total_train_iters_produced)
#         self.dataset.set_augmentation(augment_images=augment_images)
#         self.total_train_iters_produced += (self.num_of_gpus * self.batch_size * self.samples_per_iter)
#         for sample_id, sample_batched in enumerate(self.get_dataloader()):
#             yield sample_batched
#
#     def get_val_batches(self, total_batches=-1, augment_images=False):
#         """
#         Returns a validation batches data_loader
#         :param total_batches: The number of batches we want the data loader to sample
#         :param augment_images: Whether we want the images to be augmented.
#         """
#         if total_batches == -1:
#             self.dataset.data_length = self.full_data_length
#         else:
#             self.dataset.data_length['val'] = total_batches * self.dataset.batch_size
#         self.dataset.switch_set(set_name="val")
#         self.dataset.set_augmentation(augment_images=augment_images)
#         for sample_id, sample_batched in enumerate(self.get_dataloader()):
#             yield sample_batched
#
#     def get_test_batches(self, total_batches=-1, augment_images=False):
#         """
#         Returns a testing batches data_loader
#         :param total_batches: The number of batches we want the data loader to sample
#         :param augment_images: Whether we want the images to be augmented.
#         """
#         if total_batches == -1:
#             self.dataset.data_length = self.full_data_length
#         else:
#             self.dataset.data_length['test'] = total_batches * self.dataset.batch_size
#         self.dataset.switch_set(set_name='test')
#         self.dataset.set_augmentation(augment_images=augment_images)
#         for sample_id, sample_batched in enumerate(self.get_dataloader()):
#             yield sample_batched
