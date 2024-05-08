# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
import os
import random
import numpy as np
import cv2
import argparse

def Ramdom_GaussianBlur(input):
    blur_levels = [(5, 5), (7, 7), (9, 9), (11, 11),(13,13),(15,15)]  # 不同的模糊级别，每个元素是一个元组，包含了高斯核的大小
    selected_level = random.choice(blur_levels)  # 随机选择一个模糊级别
    out = cv2.GaussianBlur(input, selected_level, 0)  # 使用选定的模糊级别进行高斯模糊处理
    return out
def trans(args,input_path1,input_path2,mode):
    Original_img = cv2.imread(input_path1).astype(np.float32) / 255.0
    Mask1_img = cv2.imread(input_path2).astype(np.float32)/ 255.0
    Mask2_img = 1 - Mask1_img
    Blurred_img = Ramdom_GaussianBlur(Original_img)

    Synthesized_imgA=Mask1_img*Original_img+(1-Mask1_img)*Blurred_img
    Synthesized_imgB=Mask2_img*Original_img+ (1-Mask2_img)*Blurred_img

    result_root1 = os.path.join(os.getcwd(), args.out_dir_name,mode,'sourceA')
    result_root2 = os.path.join(os.getcwd(), args.out_dir_name,mode,'sourceB')
    result_root3 = os.path.join(os.getcwd(), args.out_dir_name,mode,'decisionmap')
    result_root4 = os.path.join(os.getcwd(), args.out_dir_name,mode,'groundtruth')

    #Synthesized_imgA save path
    save_rootA = os.path.join(result_root1,os.path.split(input_path1)[1].split('.')[0])
    #Synthesized_imgB save path
    save_rootB = os.path.join(result_root2,os.path.split(input_path1)[1].split('.')[0])
    #mask save path
    save_rootC = os.path.join(result_root3,os.path.split(input_path1)[1].split('.')[0])
    #ori save path
    save_rootD = os.path.join(result_root4,os.path.split(input_path1)[1].split('.')[0])

    cv2.imwrite(save_rootA + '.jpg', (Synthesized_imgA * 255).astype(np.uint8))
    cv2.imwrite(save_rootB + '.jpg', (Synthesized_imgB * 255).astype(np.uint8))
    cv2.imwrite(save_rootC + '.png', (Mask1_img * 255).astype(np.uint8))
    cv2.imwrite(save_rootD + '.jpg', (Original_img * 255).astype(np.uint8))


def main(args):
    dirname=args.out_dir_name
    if os.path.exists(dirname) is False:
        os.makedirs(dirname)
    if os.path.exists('{}/train'.format(dirname)) is False:
        os.makedirs('{}/train'.format(dirname))
    if os.path.exists('{}/test'.format(dirname)) is False:
        os.makedirs('{}/test'.format(dirname))
    if os.path.exists('{}/train/sourceA'.format(dirname)) is False:
        os.makedirs('{}/train/sourceA'.format(dirname))
    if os.path.exists('{}/train/sourceB'.format(dirname)) is False:
        os.makedirs('{}/train/sourceB'.format(dirname))
    if os.path.exists('{}/test/sourceA'.format(dirname)) is False:
        os.makedirs('{}/test/sourceA'.format(dirname))
    if os.path.exists('{}/test/sourceB'.format(dirname)) is False:
        os.makedirs('{}/test/sourceB'.format(dirname))
    if os.path.exists('{}/test/groundtruth'.format(dirname)) is False:
        os.makedirs('{}/test/groundtruth'.format(dirname))
    if os.path.exists('{}/train/groundtruth'.format(dirname)) is False:
        os.makedirs('{}/train/groundtruth'.format(dirname))
    if os.path.exists('{}/test/decisionmap'.format(dirname)) is False:
        os.makedirs('{}/test/decisionmap'.format(dirname))
    if os.path.exists('{}/train/decisionmap'.format(dirname)) is False:
        os.makedirs('{}/train/decisionmap'.format(dirname))

    Ground_list_name = [i for i in os.listdir(os.path.join(args.data_root,'DUTS-{}'.format(args.mode),'DUTS-{}-Mask'.format(args.mode)))if i.endswith('png')]#MASK
    Ground_list=[os.path.join(args.data_root,'DUTS-{}'.format(args.mode),'DUTS-{}-Mask'.format(args.mode),i)for i in Ground_list_name]
    Original_list=[os.path.join(args.data_root,'DUTS-{}'.format(args.mode),'DUTS-{}-Image'.format(args.mode),i.split('.')[0]+'.jpg')for i in Ground_list_name]#原图

    for i in range(len(Ground_list)):
        if args.mode=='TR':
            trans(args,Original_list[i],Ground_list[i],'train')
            print('finish no.', i + 1, 'for train datasets')
        else:
            trans(args,Original_list[i], Ground_list[i],'test')
            print('finish no.', i + 1, 'for test datasets')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make dataset for training')
    parser.add_argument('--mode', type=str,default='TE', help='TE/TR,Correspond to making training set or verification set respectively.')
    parser.add_argument('--data_root', type=str, default=r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/DUTS',help='DUTS path')
    parser.add_argument('--out_dir_name', type=str, default='DUTS_MFF')
    args = parser.parse_args()
    main(args)
