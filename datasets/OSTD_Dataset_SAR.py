
# basic imports
import argparse
import os
import math
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Patch
from matplotlib.dates import DateFormatter
from collections import namedtuple
from PIL import Image
from scipy.io import loadmat

# DL library imports
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import v2

# import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import random_split


norms = {
    'imagenet': {'mean':(0.485, 0.456, 0.406), 'std':(0.229, 0.224, 0.225)},
    'potsdam': {'mean':(0.349, 0.371, 0.347), 'std':(0.1196, 0.1164, 0.1197)},
    'potsdam_irrg': {'mean':(0.3823, 0.3625, 0.3364), 'std':(0.1172, 0.1167, 0.1203)},
    'floodnet': {'mean':(0.4159, 0.4499, 0.3466), 'std':(0.1297, 0.1197, 0.1304)},
    'vaihingen': {'mean':(0.4731, 0.3206, 0.3182), 'std':(0.1970, 0.1306, 0.1276)},
}


def rgb_to_2D_label(label):
    """
    Suply our label masks as input in RGB format. 
    Replace pixels with specific RGB values ...
    """
    Impervious = [255, 255, 255]
    Building = [0, 0, 255]
    Vegetation = [0, 255, 255]
    Tree = [0, 255, 0]
    Car = [255, 255, 0]
    Clutter = [255, 0, 0]

    label_seg = np.zeros(label.shape,dtype=np.uint8)
    label_seg [np.all(label==Impervious,axis=-1)] = 0
    label_seg [np.all(label==Building,axis=-1)] = 1
    label_seg [np.all(label==Vegetation,axis=-1)] = 2
    label_seg [np.all(label==Tree,axis=-1)] = 3
    label_seg [np.all(label==Car,axis=-1)] = 4
    label_seg [np.all(label==Clutter,axis=-1)] = 5

    # label_seg = label_seg[:,:,0]  #Just take the first channel, no need for all 3 channels
    
    return label_seg


class OSTD_Dataset_SAR(BaseDataset):
    """Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    def __init__(
            self, 
            opt, 
            split_type,
            augmentation=False, 
    ):
        # images_dir = os.path.join(opt.data_path, split_type, "image128")
        images_dir = os.path.join(opt.data_path, split_type, "sar128")
        masks_dir = os.path.join(opt.data_path, split_type, "mask128")
        self.im_ids = sorted(os.listdir(images_dir), key=self.sort_key)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.im_ids]
        self.mask_ids = sorted(os.listdir(masks_dir), key=self.sort_key)
        self.masks_fps = [os.path.join(masks_dir, mask_id) for mask_id in self.mask_ids]
        
        self.augmentation = augmentation
        self.img_ids = self.get_img_ids(images_dir, masks_dir)
        self.opt = opt

    def __getitem__(self, i):
        # image = loadmat(self.images_fps[i])['img'].transpose(2, 0, 1)
        image = loadmat(self.images_fps[i])['sar'].transpose(2, 0, 1)
        mask = loadmat(self.masks_fps[i])['map'][:, :, 0].astype(np.int64)
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        if self.augmentation:
            image, mask = self.is_aug(image, mask)

        # print("image shape", image.shape, "mask shape", mask.shape)

        # image = self.norm(image)

        return image.float(), mask, self.img_ids[i]

    def norm(self, image):
        bands, _, _ = image.shape

        # 归一化
        for i in range(bands):
            max = torch.max(image[i, :, :])
            min = torch.min(image[i, :, :])
            if max == 0 and min == 0:
                # print(" ############################## skip ############################## ")
                continue
            image[i, :, :] = (image[i, :, :] - image[i, :, :].min()) / (image[i, :, :].max()-image[i, :, :].min())

        return image

    def is_aug(self, images, mask):

        aug = v2.Compose([
                # v2.RandomResizedCrop(112, antialias=True, interpolation=InterpolationMode.BICUBIC),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                # v2.ElasticTransform(fill=2),
                # v2.RandomRotation(15, fill=2, interpolation=InterpolationMode.BILINEAR),
                # v2.RandomRotation(15),
                # v2.RandomErasing(0.5, scale=(0.02,0.2)),
                # v2.RandomAffine(degrees=180,           # 随机旋转的角度范围 (-180, 180)
                #                 translate=(0.1, 0.1),  # 随机平移的范围（水平和垂直方向，按比例）
                #                 scale=(0.8, 1.2),      # 随机缩放的范围
                #                 shear=10,              # 随机错切的角度范围 (-10, 10)
                #                 fill=2,
                #                 interpolation=InterpolationMode.BILINEAR)
                ])
        mask = mask.unsqueeze(0)

        stacked = torch.cat((images, mask), dim=0)
        stacked = aug(stacked)
        img_transformed = stacked[:-1, :, :]
        mask_transformed = stacked[-1, :, :]

        return img_transformed, mask_transformed

    def get_img_ids(self, img_dir, mask_dir):
        img_filename_list = sorted(os.listdir(img_dir), key=self.sort_key)
        mask_filename_list = sorted(os.listdir(mask_dir), key=self.sort_key)
        assert len(img_filename_list) == len(mask_filename_list)
        img_ids = [str(id.split('.')[0]) for id in mask_filename_list]
        return img_ids
    
    def get_name(self, i):
        return self.im_ids[i]
    
    def get_id_by_name(self, im_name):
        for i, name in enumerate(self.im_ids):    
            if name == im_name:
                return i
        return 
    
    def __len__(self):
        return len(self.im_ids)

    # 自定义排序键
    def sort_key(self, filename):
        # 将文件名前缀（数字部分）提取出来并转换为整数
        return int(filename.split('.')[0][5:])
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/home/icclab/Documents/lqw/DatasetMMF/OSTD", \
                        help='path were the input data is stored')
    parser.add_argument('--num_classes', type=int, default=2, help='number of semantic classes of the dataset')

    opt = parser.parse_args()

    train_dataset = OSTD_Dataset_SAR(opt, split_type='train', augmentation=False)
    val_dataset = OSTD_Dataset_SAR(opt, split_type='val', augmentation=False)
    test_dataset = OSTD_Dataset_SAR(opt, split_type='test', augmentation=False)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    
    for (ti, tm, _), (vi, vm, _), (tei, tem, _) in zip(train_dataset, val_dataset, test_dataset):
        # 3, 256, 256 | 256, 256
        # 3, 512, 512 | 512, 512
        print("ti", ti.shape, "tm", tm.shape, "vi", vi.shape, "vm", vm.shape, "tei", tei.shape, "tem", tem.shape)
        # print(np.unique(tm), np.unique(vm), np.unique(tem))
        # print(torch.max(tm), torch.max(vm), torch.max(tem))
        # print(torch.min(tm), torch.min(vm), torch.min(tem))
        break


