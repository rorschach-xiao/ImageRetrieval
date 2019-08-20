#! /home/lijiahui/anaconda3/bin/python
import cv2
import numpy as np
import os

def scaling_rotate(img):
    height,width,channel = img.shape
    roiImg = cv2.threshold(img, 255, 0, cv2.THRESH_BINARY)[1] #生成图片尺寸不变
    # print(height,width)
    scale = [0.5,0.6,0.7,0.8,0.9,1.0,1/0.9,1/0.8,1/0.7,1/0.6,2.0]
    a = np.random.choice(scale)
    b = np.random.choice(scale)
    # print(a,b)
    dstheight = int(height * a)
    dstwidth = int(width * b)
    dst = cv2.resize(img,(dstwidth,dstheight))   #随机缩放

    dst = cv2.flip(dst,np.random.choice([0,1]))  #随机旋转

    x_begin = int(width / 2 - dstwidth / 2)
    y_begin = int(height / 2 - dstheight / 2)
    if (x_begin < 0):
        x_begin = int(width*(1-1/b))
        x_end = width
    else:
        x_end = x_begin + dstwidth

    if (y_begin < 0):
        y_begin = int(height*(1-1/a))
        y_end = height
    else:
        y_end = y_begin + dstheight

    roiImg[y_begin:y_end, x_begin:x_end] = dst[:y_end-y_begin,:x_end-x_begin]

    return roiImg



root_path = 'test'   #target文件夹
#save_path = 'D://1000_Rotate//'   #存放地址
img_list = os.listdir(root_path)

for i in range(1,11):
    for file in img_list:
        try:
            path = os.path.join(root_path,file)
            img = cv2.imread(path)
            # bg = Image.open(os.path.join(bg_path,bg_list[np.random.randint(len(bg_list))]))
            dst = scaling_rotate(img)
        #     print(save_path+file.split('.')[0] + '_pic.jpg')
            cv2.imwrite(file.split('.')[0] + '_rotate'+str(i)+'.jpg',dst)
            print("已产生"+file)
        except IOError:
            pass
        continue

