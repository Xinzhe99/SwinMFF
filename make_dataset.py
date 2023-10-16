# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
import os
from PIL import Image,ImageChops
import numpy
import cv2

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

def Resize(input):
    width, height=256,256#todo need set
    img = input.resize((width,height))
    return img

def trans(input_path1,input_path2,mode):#mode=0 train mode=1 test
    Original_img = Resize(Image.open(input_path1))
    Ground_img = Resize(Image.open(input_path2))
    Blurred_img1,Blurred_img2,Blurred_img3,Blurred_img4,Blurred_img5 = GaussianBlur(Original_img)
    Mask1_img = mask1(Ground_img)
    Mask2_img = mask2(Ground_img)
    #第1个高斯模糊度
    Part_imageA1_1 = ImageChops.multiply(Blurred_img1, Mask1_img)
    Part_imageB1_1 = ImageChops.multiply(Original_img, Mask1_img)
    Part_imageA2_1 = ImageChops.multiply(Original_img, Mask2_img)
    Part_imageB2_1 = ImageChops.multiply(Blurred_img1, Mask2_img)
    Synthesized_imgA_1 = ImageChops.add(Part_imageA1_1, Part_imageA2_1)
    Synthesized_imgB_1 = ImageChops.add(Part_imageB1_1, Part_imageB2_1)
    # 第2个高斯模糊度
    Part_imageA1_2 = ImageChops.multiply(Blurred_img2, Mask1_img)
    Part_imageB1_2 = ImageChops.multiply(Original_img, Mask1_img)
    Part_imageA2_2 = ImageChops.multiply(Original_img, Mask2_img)
    Part_imageB2_2 = ImageChops.multiply(Blurred_img2, Mask2_img)
    Synthesized_imgA_2 = ImageChops.add(Part_imageA1_2, Part_imageA2_2)
    Synthesized_imgB_2 = ImageChops.add(Part_imageB1_2, Part_imageB2_2)
    #第3个高斯模糊度
    Part_imageA1_3 = ImageChops.multiply(Blurred_img3, Mask1_img)
    Part_imageB1_3 = ImageChops.multiply(Original_img, Mask1_img)
    Part_imageA2_3 = ImageChops.multiply(Original_img, Mask2_img)
    Part_imageB2_3 = ImageChops.multiply(Blurred_img3, Mask2_img)
    Synthesized_imgA_3 = ImageChops.add(Part_imageA1_3, Part_imageA2_3)
    Synthesized_imgB_3 = ImageChops.add(Part_imageB1_3, Part_imageB2_3)
    #第4个高斯模糊度
    Part_imageA1_4 = ImageChops.multiply(Blurred_img4, Mask1_img)
    Part_imageB1_4 = ImageChops.multiply(Original_img, Mask1_img)
    Part_imageA2_4 = ImageChops.multiply(Original_img, Mask2_img)
    Part_imageB2_4 = ImageChops.multiply(Blurred_img4, Mask2_img)
    Synthesized_imgA_4 = ImageChops.add(Part_imageA1_4, Part_imageA2_4)
    Synthesized_imgB_4 = ImageChops.add(Part_imageB1_4, Part_imageB2_4)
    # 第5个高斯模糊度
    Part_imageA1_5 = ImageChops.multiply(Blurred_img5, Mask1_img)
    Part_imageB1_5 = ImageChops.multiply(Original_img, Mask1_img)
    Part_imageA2_5 = ImageChops.multiply(Original_img, Mask2_img)
    Part_imageB2_5 = ImageChops.multiply(Blurred_img5, Mask2_img)
    Synthesized_imgA_5 = ImageChops.add(Part_imageA1_5, Part_imageA2_5)
    Synthesized_imgB_5 = ImageChops.add(Part_imageB1_5, Part_imageB2_5)

    result_root1 = os.path.join(os.getcwd(), 'DUTS_224',mode,'sourceB\\')#5个模糊度5张
    result_root2 = os.path.join(os.getcwd(), 'DUTS_224',mode,'sourceA\\')#5个模糊度5张
    result_root3 = os.path.join(os.getcwd(), 'DUTS_224',mode,'decisionmap\\')#决策图保存5张
    result_root4 = os.path.join(os.getcwd(), 'DUTS_224',mode,'groundtruth\\')#真实图保存5张

    #Synthesized_imgA保存路径
    save_rootA_1 = result_root1 + os.path.split(input_path1)[1].split('.')[0]
    #Synthesized_imgB保存路径
    save_rootB_1 = result_root2 + os.path.split(input_path1)[1].split('.')[0]
    #蒙版保存路径
    save_rootC_1 = result_root3 + os.path.split(input_path1)[1].split('.')[0]
    #原图保存路径
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
def main():
    # 创建文件夹
    dirname='DUTS_256'#todo need set
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

    #os.getcwd()：E:\pycharmproject\makeupdatasets_voc2012
    ori_dir_name='DUTS'
    data_root=os.path.join(os.getcwd(),ori_dir_name)#数据集位置E:\pycharmproject\makeupdatasets_voc2012\DUTS
    #读取原数据集
    Ground_list_name = [i for i in os.listdir(os.path.join(data_root,'DUTS-TR','DUTS-TR-Mask'))if i.endswith('png')]#MASK
    Ground_list=[os.path.join(data_root,'DUTS-TR','DUTS-TR-Mask',i)for i in Ground_list_name]
    Original_list=[os.path.join(data_root,'DUTS-TR','DUTS-TR-Image',i.split('.')[0]+'.jpg')for i in Ground_list_name]#原图
    # 划分train和test，如果已经是划分的，看下面
    for i in range(len(Ground_list)):
        if i<=int(len(Ground_list)*1):#todo 0.7就是 (训练：测试)=(7：3)  不需要分就是1，然后把下面的mode改了就可以了,mode决定保存的文件夹是train还是test，必须手动设置
            trans(Original_list[i],Ground_list[i],'train')
            print('finish no.', i + 1, 'for train datasets')
        else:
            trans(Original_list[i], Ground_list[i],'test')
            print('finish no.', i + 1, 'for test datasets')

if __name__ == "__main__":
    main()