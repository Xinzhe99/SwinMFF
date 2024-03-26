# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University

import torch
import torch.nn as nn
import argparse
from tools import config_model_dir
import os.path
from torchvision import transforms
import cv2
import numpy as np
from models.SwinMFF import SwinMFF
from timm.models.layers import to_2tuple
from PIL import Image

parser = argparse.ArgumentParser(description='SwinMFF')
parser.add_argument('--predict_name',default='predict')
parser.add_argument('--is_gray',type=bool,default=False)
parser.add_argument('--input_size', type=tuple, default=256,help='number of epochs to train')
parser.add_argument('--window_size', type=int, default=8,help='number of epochs to train')
parser.add_argument('--model_path',default='./checkpoint.ckpt')
parser.add_argument('--dataset_path',default='./assets/Lytro')
parser.add_argument('--tile', type=int, default=256, help='Tile size, None for no tile during testing (testing as a whole)')
parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8')
args = parser.parse_args()

predict_save_path=config_model_dir(subdir_name='run')
single_path = os.path.join(predict_save_path, 'y')
fuse_path=os.path.join(predict_save_path, 'result')

if not os.path.exists(single_path):
    os.makedirs(single_path)
if not os.path.exists(fuse_path):
    os.makedirs(fuse_path)

model = SwinMFF(img_size=to_2tuple(args.input_size),window_size=args.window_size)
model=nn.DataParallel(model)
if torch.cuda.device_count() > 1:
    model.cuda()
model.load_state_dict(torch.load(args.model_path,map_location=lambda storage, loc: storage)['model'],strict=False)
model.eval()

transform=transforms.Compose([transforms.ToTensor()])

sub_dirs = os.listdir(args.dataset_path)
sub_dirs.sort()
dir_A = os.path.join(args.dataset_path, sub_dirs[0])
dir_B = os.path.join(args.dataset_path, sub_dirs[1])
img_names = os.listdir(dir_A)
img_names.sort(key=lambda x: int(x[:-4]))
num = len(img_names)

#fusion y
def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img * 255.0).round())

for i in range(num):
    img0 = os.path.join(dir_A, img_names[i])
    img1 = os.path.join(dir_B, img_names[i])

    img0_Y=np.array(Image.open(img0).convert('YCbCr').split()[0])
    img1_Y=np.array(Image.open(img1).convert('YCbCr').split()[0])

    ori_h=img0_Y.shape[0]
    ori_w=img0_Y.shape[1]

    if min(ori_h,ori_w)<args.tile:
        # If both width and height are less than 256, fill both sides to 256
        if ori_h < 256 and ori_w < 256:
            img0_Y = cv2.copyMakeBorder(img0_Y, 0, 256 - img0_Y.shape[0], 0, 256 - img0_Y.shape[1], cv2.BORDER_CONSTANT, value=0)
            img1_Y = cv2.copyMakeBorder(img1_Y, 0, 256 - img1_Y.shape[0], 0, 256 - img1_Y.shape[1], cv2.BORDER_CONSTANT, value=0)
        # If the width is less than 256, only fill the wide edge
        elif ori_w < 256:
            img0_Y = cv2.copyMakeBorder(img0_Y, 0, 0, 0, 256 - img0_Y.shape[1], cv2.BORDER_CONSTANT, value=0)
            img1_Y = cv2.copyMakeBorder(img1_Y, 0, 0, 0, 256 - img1_Y.shape[1], cv2.BORDER_CONSTANT, value=0)
        # If the height is less than 256, only the high side will be filled.
        elif ori_h < 256:
            img0_Y = cv2.copyMakeBorder(img0_Y, 0, 256 - img0_Y.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0)
            img1_Y = cv2.copyMakeBorder(img1_Y, 0, 256 - img1_Y.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0)

    img0_Y_tensor = transform(img0_Y)
    img1_Y_tensor = transform(img1_Y)

    img0_Y_tensor.unsqueeze_(0)
    img1_Y_tensor.unsqueeze_(0)

    if torch.cuda.is_available():
        img0_Y_tensor = img0_Y_tensor.cuda()
        img1_Y_tensor = img1_Y_tensor.cuda()
    else:
        img0_Y_tensor = img0_Y_tensor
        img1_Y_tensor = img1_Y_tensor

    tile=args.tile
    stride = args.tile-args.tile_overlap
    b, c, h, w = img0_Y_tensor.shape

    h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
    w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
    out = torch.zeros(b, c, h, w).cuda()

    sf = args.scale
    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch_0 = img0_Y_tensor[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
            in_patch_1 = img1_Y_tensor[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
            out_patch = model(in_patch_0,in_patch_1)
            out[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf]=out_patch

    # If it has been filled, crop it out
    if min(ori_h,ori_w)<args.tile:
        output = np.squeeze(tensor2uint(out)[:ori_h, :ori_w])#todo
    else:
        output = np.squeeze(tensor2uint(out))
    cv2.imwrite(os.path.join(single_path, img_names[i]), output)
    if args.is_gray==True:
        print(os.path.join(single_path, img_names[i]),' finish generate gray!')
    else:
        print(os.path.join(single_path, img_names[i]), ' finish generate Y channel!')

#If it is not a grayscale image, continue to fuse the RGB image
if not args.is_gray:
#fusion pic
    def RGB2YCbCr(img_rgb):
        R = img_rgb[:, :, 0]
        G = img_rgb[:, :, 1]
        B = img_rgb[:, :, 2]
        # RGB to YCbCr
        Y = 0.257 * R + 0.564 * G + 0.098 * B + 16
        Cb = -0.148 * R - 0.291 * G + 0.439 * B + 128
        Cr = 0.439 * R - 0.368 * G - 0.071 * B + 128
        return Y, Cb, Cr

    def YCbCr2RGB(img_YCbCr):
        Y = img_YCbCr[:, :, 0]
        Cb = img_YCbCr[:, :, 1]
        Cr = img_YCbCr[:, :, 2]
        # YCbCr to RGB
        R = 1.164 * (Y - 16) + 1.596 * (Cr - 128)
        G = 1.164 * (Y - 16) - 0.392 * (Cb - 128) - 0.813 * (Cr - 128)
        B = 1.164 * (Y - 16) + 2.017 * (Cb - 128)
        image_RGB = np.dstack((R, G, B))
        return image_RGB

    for i in range(num):
        name_A = os.path.join(dir_A, img_names[i])
        name_B = os.path.join(dir_B, img_names[i])
        name_fused = os.path.join(single_path, img_names[i])
        save_name = os.path.join(fuse_path, img_names[i])

        image_A = cv2.imread(name_A)
        image_B = cv2.imread(name_B)
        I_result = cv2.imread(name_fused,cv2.IMREAD_UNCHANGED)

        Y1, Cb1, Cr1 = RGB2YCbCr(image_A)
        Y2, Cb2, Cr2 = RGB2YCbCr(image_B)

        H, W = Cb1.shape
        Cb = np.ones((H, W))
        Cr = np.ones((H, W))

        for k in range(H):
            for n in range(W):
                if abs(Cb1[k, n] - 128) == 0 and abs(Cb2[k, n] - 128) == 0:
                    Cb[k, n] = 128
                else:
                    middle_1 = Cb1[k, n] * abs(Cb1[k, n] - 128) + Cb2[k, n] * abs(Cb2[k, n] - 128)
                    middle_2 = abs(Cb1[k, n] - 128) + abs(Cb2[k, n] - 128)
                    Cb[k, n] = middle_1 / middle_2

                if abs(Cr1[k, n] - 128) == 0 and abs(Cr2[k, n] - 128) == 0:
                    Cr[k, n] = 128
                else:
                    middle_3 = Cr1[k, n] * abs(Cr1[k, n] - 128) + Cr2[k, n] * abs(Cr2[k, n] - 128)
                    middle_4 = abs(Cr1[k, n] - 128) + abs(Cr2[k, n] - 128)
                    Cr[k, n] = middle_3 / middle_4

        I_final_YCbCr = np.dstack((I_result, Cb, Cr))
        I_final_RGB = YCbCr2RGB(I_final_YCbCr)
        cv2.imwrite(save_name, I_final_RGB)
        print(save_name,' finish fusion!')
