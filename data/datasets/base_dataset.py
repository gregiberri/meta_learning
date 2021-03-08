# -*- coding: utf-8 -*-
# @Time    : 2019/12/25 下午5:09
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com
# @File    : base_dataset.py

import os
import torch
import numpy as np

import torch.utils.data as data


class BaseDataset(data.Dataset):
    def __init__(self, config, is_train=True, image_loader=None, depth_loader=None):
        super(BaseDataset, self).__init__()
        self.is_train = is_train
        self.config = config
        self.dataset_path = self.config.path
        self.split = self.config.split
        self.split = self.split[0] if is_train else self.split[1]
        self.image_loader, self.depth_loader = image_loader, depth_loader
        self.lidar_sparsity = 1.
        self.lidar_sparsity_decay = config.lidar_sparsity_decay

        self.preprocess = self._preprocess

    def _get_filepaths(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.filenames['image_paths'])

    def __getitem__(self, index):
        image_path = self.filenames['image_paths'][index]
        depth_path = self.filenames['depth_paths'][index]
        gt_path = self.filenames['gt_paths'][index]
        item_name = image_path.split("/")[-1].split(".")[0]
        scene_name = image_path.split("/")[-4]

        image, depth, gt = self._fetch_data(image_path, depth_path, gt_path)
        image, depth, gt, extra_dict = self.preprocess(image, depth, gt)
        image = torch.from_numpy(np.ascontiguousarray(image)).float()
        depth = torch.from_numpy(np.ascontiguousarray(depth)).float()

        output_dict = dict(image=image,
                           depth=depth,
                           fn=str(item_name),
                           image_path=image_path,
                           n=self.get_length(),
                           scene=str(scene_name))

        if gt is not None:
            output_dict['target'] = torch.from_numpy(np.ascontiguousarray(gt)).float()
            output_dict['target_path'] = gt_path

        if extra_dict is not None:
            output_dict.update(**extra_dict)

        return output_dict

    def _fetch_data(self, image_path, depth_path=None, gt_path=None):
        image = self.image_loader(image_path)
        gt = None
        depth = None
        if gt_path is not None:
            gt = self.depth_loader(gt_path)
        if depth_path is not None:
            depth = self.depth_loader(depth_path)
        return image, depth, gt

    def get_length(self):
        return self.__len__()

    def _preprocess(self, image, depth, gt):
        raise NotImplementedError
