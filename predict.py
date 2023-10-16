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
from net import SwinMFF_Net
from timm.models.layers import to_2tuple
from Dataloader import get_transform
parser = argparse.ArgumentParser(description='SwinMFIF-Fusion')
parser.add_argument('--predict_name',default='predict')
parser.add_argument('--input_size', type=tuple, default=520,help='number of epochs to train')
parser.add_argument('--model_path',default='/home/oceanthink/xxz/SwinMFIF_New/SwinMFIF_Test_runs/SwinMFIF_Test_runs51/checkpoint_29.ckpt')
parser.add_argument('--img1_path',default='./sample_data/lytro-01-A.jpg')
parser.add_argument('--img2_path',default='./sample_data/lytro-01-B.jpg')
parser.add_argument('--format',default='.jpg')

args = parser.parse_args()

img0 = cv2.imread(args.img1_path)
img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
img0 = cv2.resize(img0, (args.input_size, args.input_size), interpolation=cv2.INTER_AREA)

img1 = cv2.imread(args.img2_path)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img1 = cv2.resize(img1, (args.input_size, args.input_size), interpolation=cv2.INTER_AREA)

predict_save_path=config_model_dir(subdir_name='predict_runs')
print(predict_save_path)
model = SwinMFF_Net(img_size=to_2tuple(args.input_size),window_size=13)
model=nn.DataParallel(model)

if torch.cuda.device_count() > 1:
    model.cuda()

model.load_state_dict(torch.load(args.model_path,map_location=lambda storage, loc: storage)['model'],strict=False)
model.eval()

def tensor2np2color(intensor):
    # trans = transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    # intensor=trans(intensor)
    out = intensor[0].cpu().detach().numpy().transpose(1, 2, 0)  # + 1) / 2)
    out = out* 255.0
    # out = (out - np.min(out)) / (np.max(out) - np.min(out)) * 255.0  # 将图像数据扩展到[0,255]
    out = np.array(out, dtype='uint8')  # 改为Unit8
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return out


processed = get_transform()

img0 = processed(img0)
img1 = processed(img1)
img0.unsqueeze_(0)
img1.unsqueeze_(0)
if torch.cuda.is_available():
    img0 = img0.cuda()
    img1 = img1.cuda()
else:
    img0 = img0
    img1 = img1

out=model(img0,img1)
# out=torch.clamp(out,0,1)
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
#
# out[:, 0, :, :] = out[:, 0, :, :] * std[0] + mean[0]  # R channel
# out[:, 1, :, :] = out[:, 1, :, :] * std[1] + mean[1]  # G channel
# out[:, 2, :, :] = out[:, 2, :, :] * std[2] + mean[2]  # B channel

out_image = transforms.ToPILImage()(torch.squeeze(out.data.cpu(), 0))
out_image.save(os.path.join(predict_save_path, args.predict_name+args.format))
#
# out=tensor2np2color(out)
# cv2.imwrite(os.path.join(predict_save_path, args.predict_name+args.format),out)


