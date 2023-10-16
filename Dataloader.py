# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
import glob
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath,mode='end2end'):

    all_left_img=glob.glob(filepath+'/train/sourceA/*.jpg')
    all_right_img = glob.glob(filepath + '/train/sourceB/*.jpg')
    test_left_img =glob.glob(filepath+'/test/sourceA/*.jpg')
    test_right_img = glob.glob(filepath + '/test/sourceB/*.jpg')
    if mode=='end2end':

        label_train_img =glob.glob(filepath+'/train/groundtruth/*.jpg')
        label_val_img=glob.glob(filepath+'/test/groundtruth/*.jpg')
    elif mode=='decisionmap':
        label_train_img = glob.glob(filepath + '/train/decisionmap/*.png')
        label_val_img = glob.glob(filepath + '/test/decisionmap/*.png')
    else:
        print('mode= end2end or decisionmap! Please check!')
    return all_left_img, all_right_img, test_left_img, test_right_img, label_train_img,label_val_img

def default_loader(path,mode='y'):
    if mode=='y':
        return np.array(Image.open(path).convert('YCbCr').split()[0])#yCbCr
    elif mode=='dm':
        return np.array(Image.open(path).convert('L'))
    else:
        print('mode= y or dm! Please check!')
def get_transform(seed=None,augment=False):

    if augment:
        seed = seed
        random.seed(seed)
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.33),
            transforms.RandomVerticalFlip(p=0.33),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
        ])
class myImageFloder(data.Dataset):
    def __init__(self, left, right,label_img,augment,mode,loader=default_loader):
        self.left = left
        self.right = right
        self.label_img = label_img
        self.augment=augment
        self.loader = loader
        self.mode=mode
    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        label_img = self.label_img[index]

        left_img = self.loader(left,mode='y')  # PIL read
        right_img = self.loader(right,mode='y')  # PIL read
        if self.mode=='end2end':
            label_img = self.loader(label_img, mode='y')
        elif self.mode=='decisionmap':
            label_img = self.loader(label_img,mode='y')
        seed = np.random.randint(666666)

        processed = get_transform(augment=self.augment,seed=seed)

        left_img = processed(left_img)
        right_img = processed(right_img)
        label_img = processed(label_img)


        return left_img, right_img, label_img

    def __len__(self):
        return len(self.left)
