# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
import os.path
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage import io
from tqdm import tqdm

gt_path=r'Ground Truth Path'
methods_root=r'method root'
# compare_methods=['IFCNN-MAX','SDNet','MFF-GAN','SwinFusion','U2Fusion','SwinMFF']
compare_methods=['U2Fusion']
psnr_all=0
ssim_all=0
mse_all=0

cal_all=True

index=79

# 读取原始图像和处理后的图像
for method in compare_methods:
    psnr_all = 0
    ssim_all = 0
    mse_all = 0
    if cal_all:
        for i in tqdm(range(len(os.listdir(gt_path)))):
            image_path=os.path.join(methods_root,method)
            original_image = io.imread(os.path.join(gt_path,str(i+1)+'.jpg'))
            processed_image = io.imread(os.path.join(image_path,'y',str(i+1)+'.jpg'))
            # 计算PSNR
            psnr_value = psnr(original_image, processed_image)
            # 计算SSIM
            ssim_value = ssim(original_image, processed_image, multichannel=True)
            mse_value=mse(original_image, processed_image)
            psnr_all+=psnr_value
            ssim_all+=ssim_value
            mse_all+=mse_value
        print(method)
        print('psnr:',psnr_all/len(os.listdir(gt_path)))
        print('ssim:',ssim_all/len(os.listdir(gt_path)))
        print('mse',mse_all/len(os.listdir(gt_path)))
    else:
        image_path = os.path.join(methods_root, method)
        original_image = io.imread(os.path.join(gt_path, str(index) + '.jpg'))
        processed_image = io.imread(os.path.join(image_path, 'y', str(index) + '.jpg'))
        # 计算PSNR
        psnr_value = psnr(original_image, processed_image)
        # 计算SSIM
        ssim_value = ssim(original_image, processed_image, multichannel=True)
        mse_value = mse(original_image, processed_image)
        print(method)
        print('psnr:', psnr_value)
        print('ssim:',ssim_value)
        print('mse', mse_value)
