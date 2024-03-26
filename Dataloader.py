# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
import glob
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
def dataloader(filepath):

    all_left_img=glob.glob(filepath+'/train/sourceA/*.jpg')
    all_right_img = glob.glob(filepath + '/train/sourceB/*.jpg')
    test_left_img =glob.glob(filepath+'/test/sourceA/*.jpg')
    test_right_img = glob.glob(filepath + '/test/sourceB/*.jpg')
    label_train_img =glob.glob(filepath+'/train/groundtruth/*.jpg')
    label_val_img=glob.glob(filepath+'/test/groundtruth/*.jpg')

    return all_left_img, all_right_img, test_left_img, test_right_img, label_train_img,label_val_img

def default_loader(path):
    return np.array(Image.open(path).convert('YCbCr').split()[0])#yCbCr
def get_transform(augment=False):
    if augment:
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
    def __init__(self, left, right,label_img,augment,loader=default_loader):
        self.left = left
        self.right = right
        self.label_img = label_img
        self.augment=augment
        self.loader = loader
    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        label_img = self.label_img[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        label_img = self.loader(label_img)
        processed = get_transform(augment=self.augment)

        left_img = processed(left_img)
        right_img = processed(right_img)
        label_img = processed(label_img)

        return left_img, right_img, label_img

    def __len__(self):
        return len(self.left)
