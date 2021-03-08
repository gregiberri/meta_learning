from torch.utils.data import DataLoader

from data.datasets import mini_imagenet, cifar_100
import numpy as np


def get_dataloader(data_config, current_set):
    # get the iterator object
    if data_config.name == 'imagenet':
        raise NotImplementedError()
    elif data_config.name == 'mini_imagenet':
        dataset = mini_imagenet.MiniImagenetDataset(data_config.params, current_set)
    elif data_config.name == 'cifar_100':
        dataset = cifar_100.CIFAR100Dataset(data_config.params, current_set)
    elif data_config.name == 'carla':
        raise NotImplementedError()
    else:
        raise ValueError(f'Wrong dataset name: {data_config.name}')

    # calculate the iteration number for the tqdm
    batch_size = data_config.params.batch_size if current_set else 1
    niters_per_epoch = int(np.ceil(dataset.data_length // batch_size))

    # make the torch dataloader object
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=data_config.params.workers,
                        drop_last=False,
                        shuffle=current_set == 'train',
                        pin_memory=False)

    return loader, niters_per_epoch
