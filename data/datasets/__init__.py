from torch.utils.data import DataLoader

from data.datasets import mini_imagenet, cifar_100, imagenet
import numpy as np


def get_dataloader(data_config, mode):
    # get the iterator object
    if data_config.name == 'imagenet':
        dataset = imagenet.ImagenetDataset(data_config.params, mode)
    elif data_config.name == 'mini_imagenet':
        dataset = mini_imagenet.MiniImagenetDataset(data_config.params, mode)
    elif data_config.name == 'cifar_100':
        dataset = cifar_100.CIFAR100Dataset(data_config.params, mode)
    elif data_config.name == 'carla':
        raise NotImplementedError()
    else:
        raise ValueError(f'Wrong dataset name: {data_config.name}')

    # calculate the iteration number for the tqdm
    batch_size = data_config.params.batch_size if 'train' in mode else 1

    # make the torch dataloader object
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=data_config.params.workers,
                        drop_last=False,
                        shuffle='train' in mode,
                        pin_memory=False)

    return loader
