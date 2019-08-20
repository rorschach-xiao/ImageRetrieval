#! /home/lijiahui/anaconda3/bin/python
import datetime
import glob
from time import sleep
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import matplotlib.pyplot as plt
import os
import random
import numpy as np



def add_watermark(imageFile, watermark):
    dst = watermark.rotate(random.randint(0, 360))
    dst2 = dst.resize((random.randint(50, 99),random.randint(50, 99)))
    layer = Image.new('RGBA', imageFile.size, (0, 0, 0, 0))
    layer.paste(dst2, (imageFile.size[0] - random.randint(30,layer.size[0]), imageFile.size[1] - random.randint(30, layer.size[1])))
    out = Image.composite(layer, imageFile, layer)
    return out


if __name__=='__main__':
    root_path = 'test'  # target文件夹
    water_path = 'WaterMark'  # 水印图片文件夹
#    save_path = 'D://1000_WaterMark//'  # 存放地址
for i in range(1,11):
    img_list = os.listdir(root_path)
    water_list = os.listdir(water_path)
    for file in img_list:
        try:
            path = os.path.join(root_path, file)
            imageFile = Image.open(path)
            watermark = Image.open(os.path.join(water_path, water_list[np.random.randint(len(water_list))]))
            out = add_watermark(imageFile, watermark)
            out.save(file.split(".")[0] + "_WaterMark_"+str(i)+".jpg")
            print("完成"+file)
        except OSError:
            pass
        continue
