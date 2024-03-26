# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
import os
from PIL import Image,ImageChops
import numpy
import cv2
import argparse
def GaussianBlur(input):
    out1 = Image.fromarray(cv2.GaussianBlur(numpy.array(input), (7, 7), 2))
    out2 = Image.fromarray(cv2.GaussianBlur(numpy.array(out1), (7, 7), 2))
    out3 = Image.fromarray(cv2.GaussianBlur(numpy.array(out2), (7, 7), 2))
    out4 = Image.fromarray(cv2.GaussianBlur(numpy.array(out3), (7, 7), 2))
    out5 = Image.fromarray(cv2.GaussianBlur(numpy.array(out4), (7, 7), 2))
    return out1, out2, out3, out4, out5

def mask1(input):
    img = input.convert('RGB')
    for x in range(img.width):
        for y in range(img.height):
            data = img.getpixel((x, y))
            if data[0]+data[1]+data[2]!=0:
                img.putpixel((x, y), (255,255,255))
    return img

def mask2(input):
    img = input.convert('RGB')
    for x in range(img.width):
        for y in range(img.height):
            data = img.getpixel((x, y))
            if data[0]+data[1]+data[2] == 0:
                img.putpixel((x, y), (255,255,255))
            else:
                img.putpixel((x, y), (0, 0, 0))
    return img

def Resize(args,input):
    width, height=args.img_size,args.img_size#todo need set
    img = input.resize((width,height))
    return img

def trans(args,input_path1,input_path2,mode):
    Original_img = Resize(args,Image.open(input_path1))
    Ground_img = Resize(args,Image.open(input_path2))
    Blurred_img1,Blurred_img2,Blurred_img3,Blurred_img4,Blurred_img5 = GaussianBlur(Original_img)
    Mask1_img = mask1(Ground_img)
    Mask2_img = mask2(Ground_img)
    # no.1 gas level
    Part_imageA1_1 = ImageChops.multiply(Blurred_img1, Mask1_img)
    Part_imageB1_1 = ImageChops.multiply(Original_img, Mask1_img)
    Part_imageA2_1 = ImageChops.multiply(Original_img, Mask2_img)
    Part_imageB2_1 = ImageChops.multiply(Blurred_img1, Mask2_img)
    Synthesized_imgA_1 = ImageChops.add(Part_imageA1_1, Part_imageA2_1)
    Synthesized_imgB_1 = ImageChops.add(Part_imageB1_1, Part_imageB2_1)
    # no.2 gas level
    Part_imageA1_2 = ImageChops.multiply(Blurred_img2, Mask1_img)
    Part_imageB1_2 = ImageChops.multiply(Original_img, Mask1_img)
    Part_imageA2_2 = ImageChops.multiply(Original_img, Mask2_img)
    Part_imageB2_2 = ImageChops.multiply(Blurred_img2, Mask2_img)
    Synthesized_imgA_2 = ImageChops.add(Part_imageA1_2, Part_imageA2_2)
    Synthesized_imgB_2 = ImageChops.add(Part_imageB1_2, Part_imageB2_2)
    # no.3 gas level
    Part_imageA1_3 = ImageChops.multiply(Blurred_img3, Mask1_img)
    Part_imageB1_3 = ImageChops.multiply(Original_img, Mask1_img)
    Part_imageA2_3 = ImageChops.multiply(Original_img, Mask2_img)
    Part_imageB2_3 = ImageChops.multiply(Blurred_img3, Mask2_img)
    Synthesized_imgA_3 = ImageChops.add(Part_imageA1_3, Part_imageA2_3)
    Synthesized_imgB_3 = ImageChops.add(Part_imageB1_3, Part_imageB2_3)
    # no.4 gas level
    Part_imageA1_4 = ImageChops.multiply(Blurred_img4, Mask1_img)
    Part_imageB1_4 = ImageChops.multiply(Original_img, Mask1_img)
    Part_imageA2_4 = ImageChops.multiply(Original_img, Mask2_img)
    Part_imageB2_4 = ImageChops.multiply(Blurred_img4, Mask2_img)
    Synthesized_imgA_4 = ImageChops.add(Part_imageA1_4, Part_imageA2_4)
    Synthesized_imgB_4 = ImageChops.add(Part_imageB1_4, Part_imageB2_4)
    # no.5 gas level
    Part_imageA1_5 = ImageChops.multiply(Blurred_img5, Mask1_img)
    Part_imageB1_5 = ImageChops.multiply(Original_img, Mask1_img)
    Part_imageA2_5 = ImageChops.multiply(Original_img, Mask2_img)
    Part_imageB2_5 = ImageChops.multiply(Blurred_img5, Mask2_img)
    Synthesized_imgA_5 = ImageChops.add(Part_imageA1_5, Part_imageA2_5)
    Synthesized_imgB_5 = ImageChops.add(Part_imageB1_5, Part_imageB2_5)

    result_root1 = os.path.join(os.getcwd(), 'DUTS_{}'.format(str(args.img_size)),mode,'sourceB\\')#5个模糊度5张
    result_root2 = os.path.join(os.getcwd(), 'DUTS_{}'.format(str(args.img_size)),mode,'sourceA\\')#5个模糊度5张
    result_root3 = os.path.join(os.getcwd(), 'DUTS_{}'.format(str(args.img_size)),mode,'decisionmap\\')#决策图保存5张
    result_root4 = os.path.join(os.getcwd(), 'DUTS_{}'.format(str(args.img_size)),mode,'groundtruth\\')#真实图保存5张

    #Synthesized_imgA save path
    save_rootA_1 = result_root1 + os.path.split(input_path1)[1].split('.')[0]
    #Synthesized_imgB save path
    save_rootB_1 = result_root2 + os.path.split(input_path1)[1].split('.')[0]
    #mask save path
    save_rootC_1 = result_root3 + os.path.split(input_path1)[1].split('.')[0]
    #ori save path
    save_rootD_1 = result_root4 + os.path.split(input_path1)[1].split('.')[0]

    Synthesized_imgA_1.save(save_rootA_1 + '_1.jpg')
    Synthesized_imgA_2.save(save_rootA_1 + '_2.jpg')
    Synthesized_imgA_3.save(save_rootA_1 + '_3.jpg')
    Synthesized_imgA_4.save(save_rootA_1 + '_4.jpg')
    Synthesized_imgA_5.save(save_rootA_1 + '_5.jpg')

    Synthesized_imgB_1.save(save_rootB_1 + '_1.jpg')
    Synthesized_imgB_2.save(save_rootB_1 + '_2.jpg')
    Synthesized_imgB_3.save(save_rootB_1 + '_3.jpg')
    Synthesized_imgB_4.save(save_rootB_1 + '_4.jpg')
    Synthesized_imgB_5.save(save_rootB_1 + '_5.jpg')

    Mask1_img.save(save_rootC_1+'_1.png')
    Mask1_img.save(save_rootC_1 + '_2.png')
    Mask1_img.save(save_rootC_1 + '_3.png')
    Mask1_img.save(save_rootC_1 + '_4.png')
    Mask1_img.save(save_rootC_1 + '_5.png')

    Original_img.save(save_rootD_1 + '_1.jpg')
    Original_img.save(save_rootD_1 + '_2.jpg')
    Original_img.save(save_rootD_1 + '_3.jpg')
    Original_img.save(save_rootD_1 + '_4.jpg')
    Original_img.save(save_rootD_1 + '_5.jpg')
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
    parser.add_argument('--mode', type=str,default='TR', help='TE/TR,Correspond to making training set or verification set respectively.')
    parser.add_argument('--data_root', type=str, default='./DUTS',help='DUTS path')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--out_dir_name', type=str, default='DUTS_256')
    args = parser.parse_args()
    main(args)