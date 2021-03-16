#!/usr/bin/env python

import collections
import os.path as osp

import numpy as np
import PIL.Image
import scipy.io
import torch
from torch.utils import data


class DeepFashionDataset(data.Dataset):

    class_names = None
    mean_bgr = np.array([187.4646117, 190.3556895, 198.6592035])

    def __init__(self, root, split='train', transform=False):
        self.root = root
        self.split = split
        self._transform = transform  # train, val都为True

        lines = open('../../torchfcn/datasets/dp_train_val.txt').readlines()
        self.files = collections.defaultdict(list)
        for l in lines:
            sub_path, tag = l.split()[0], l.split()[-1]
            image_path = osp.join(self.root, 'img_320_512_image', sub_path)
            label_path = osp.join(self.root, 'img_320_512_parsing', sub_path).replace('.jpg', '_gray.png')
            self.files[tag].append({
                'img': image_path,
                'lbl': label_path,
            })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load label
        lbl_file = data_file['lbl']
        lbl = PIL.Image.open(lbl_file)  # 'JpegImageFile' object
        lbl = np.array(lbl, dtype=np.int8)  # 原为32？
        # lbl[lbl == 255] = -1
        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl

    def transform(self, img, lbl):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)  # CHW -> HWC
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]  # RGB <- BGR
        lbl = lbl.numpy()
        return img, lbl


