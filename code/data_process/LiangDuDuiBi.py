#! /home/lijiahui/anaconda3/bin/python
import cv2
import numpy as np
import os
import matplotlib
import random

if __name__ == "__main__":
    root_path = 'test'  # target文件夹
    #    save_path = 'result'  # 存放地址
    files = os.listdir(root_path)

for i in range(1,11):
    for file in files:
        try:
            path = os.path.join(root_path, file)
            img = cv2.imread(path)
            # b控制亮度，a、b控制对比度
            a=random.randint(10,30)*0.1
            b=random.randint(10,100)
            res = np.uint8(np.clip((a * img + b), 0, 255))
            cv2.imwrite(file.split('.')[0] + "_LiangDuDuiBi_"+str(i)+".jpg", res * 255)
            print(file + "已产生")
        except TypeError:
            pass
        continue
