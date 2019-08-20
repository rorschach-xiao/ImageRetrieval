#! /home/lijiahui/anaconda3/bin/python
import numpy as np
from PIL import Image
import os
import cv2

def pic_in_pic(img, bg):
    high, widh, deep = img.shape
    height, width, channel = bg.shape
    a = np.random.uniform(0.01, 1)
    b = np.random.uniform(0.01, 1)
    dstheight = int(high * a)
    dstwidth = int(widh * b)
    # print(dstwidth,dstheight)
    dst = cv2.resize(img, (dstwidth, dstheight))
    x_begin = int(width / 2 - dstwidth / 2)
    y_begin = int(height / 2 - dstheight / 2)
    if (x_begin <= 0):
        x_begin = int(width * 0.2)
        x_end = width
    else:
        x_end = x_begin + dstwidth

    if (y_begin <= 0):
        y_begin = int(height * 0.2)
        y_end = height
    else:
        y_end = y_begin + dstheight

    bg[y_begin:y_end, x_begin:x_end] = dst[:y_end - y_begin, :x_end - x_begin]
    return bg

num = 11
root_path = 'test'   #target文件夹
bg_path = 'BackGround'  #随机图片文件夹
#save_path = 'D://yw//shengc//'   #存放地址
img_list = os.listdir(root_path)
bg_list = os.listdir(bg_path)
# print(np.random.randint(len(bg_list)))

for i in range(1, num):
    for file in img_list:
        path = os.path.join(root_path, file)
        img = cv2.imread(path)
        bg = cv2.imread(os.path.join(bg_path, bg_list[np.random.randint(len(bg_list))]))
        if bg is None:  # 判断读入的bg是否为空
            bg = cv2.imread('BackGround/DB_18117.jpg')
        # print(img.shape)
        fi = pic_in_pic(img, bg)
        print(file + '_finished')
        cv2.imwrite(file.split('.')[0] + '_' + str(i) + '_pic.jpg', fi)
