import os
import matplotlib.pyplot as plt
from PIL import Image
import re
#非resume模式下 创建新的增量式文件夹以保存文件 resume模式下则继续上一个文件夹 返回模型保存的文件夹路径
def config_model_dir(resume=False,subdir_name='train_runs'):
    # 获取当前项目目录
    project_dir = os.getcwd()
    # 获取models文件夹的路径
    models_dir = os.path.join(project_dir, subdir_name)
    # 如果models文件夹不存在，就创建它
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    # 如果是第一次运行 创建models0
    if not os.path.exists(os.path.join(models_dir,subdir_name+'1')):
        os.mkdir(os.path.join(models_dir,subdir_name+'1'))
        return os.path.join(models_dir,subdir_name+'1')
    else:
        # 获取models文件夹下已有的子文件夹
        sub_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
        sub_dirs.sort(key=lambda l: int(re.findall('\d+', l)[0]))
        last_numbers=re.findall("\d+",sub_dirs[-1])#list
        if resume==False:
            new_sub_dir_name = subdir_name + str(int(last_numbers[0]) + 1)
        else:
            new_sub_dir_name = subdir_name + str(int(last_numbers[0]))
        model_dir_path = os.path.join(models_dir, new_sub_dir_name)
        if resume == False:
            os.mkdir(model_dir_path)
        else:
            pass
        # 把创建的路径存在model_save变量里
        print(model_dir_path)
        return model_dir_path
#画loss图
def plt_train_process(y1,y2,train_len,test_len):
    x1 = range(0, train_len)
    x2 = range(0, test_len)
    plt.plot(x1, y1)
    plt.plot(x2, y2)
    plt.title('train_loss vs val_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(["train", "val"], shadow=True, fancybox="blue")
    return plt
#转伪彩色
def pred2color(pred):#in numpy out:pil
    from PIL import Image
    import cv2
    import numpy as np
    # 用PIL打开灰度图并转换为numpy数组
    img = np.array(Image.open(pred))
    # 用opencv应用彩色映射（注意opencv使用BGR通道顺序）
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

#图片转xx的倍数
def img_resize2_x(img,x=16):
    w, h = img.size
    h1 = h % x
    w1 = w % x
    h1 = h  - h1
    w1 =  w - w1
    h1 = int(h1)
    w1 = int(w1)
    img_16x = img.resize((w1, h1),Image.ANTIALIAS)
    return img_16x

