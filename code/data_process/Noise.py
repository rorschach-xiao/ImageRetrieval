#! /home/lijiahui/anaconda3/bin/python
#批量产生经不同强度的高斯噪声和椒盐噪声处理后的图片

import os
import numpy as np
from skimage import util
import cv2

def add_noise(img):
    if (np.random.choice([0, 1])):
        noise_img = util.random_noise(img, mode='gaussian', var=np.random.uniform(0, 1))  # gaussian高斯白噪声

    else:
        noise_img = util.random_noise(img, mode='s&p', amount=np.random.uniform(0, 1))  # s&p椒盐噪声

    return noise_img


if __name__ == "__main__":
    root_path = 'test'  # target文件夹
#    save_path = 'D://1000_Noise//'  # 存放地址
    img_list = os.listdir(root_path)

for i in range(1,11):
    for file in img_list:
        try:
            path = os.path.join(root_path, file)
            img = cv2.imread(path)
            noise_img = add_noise(img)
            cv2.imwrite(file.split('.')[0] + "_noise_"+str(i)+".jpg", noise_img * 255)
            print("已产生"+file)
        except IOError:
            pass
        continue






