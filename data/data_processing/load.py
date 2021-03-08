# -*- coding: utf-8 -*-
# @Time    : 2020/1/1 下午8:08
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com
# @File    : prob.py

import os
import numpy as np
from PIL import Image


def kitti_image_loader(file):
    return np.array(Image.open(file).convert('RGB'), np.uint8)


def kitti_depth_loader(file):
    # loads depth map D from png file
    assert os.path.exists(file), "file not found: {}".format(file)
    depth_png = np.array(Image.open(file), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert (np.max(depth_png) > 255)

    depth = depth_png.astype(np.float32) / 256.
    depth[depth_png == 0] = -1.
    return depth
