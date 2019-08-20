#! /home/lijiahui/anaconda3/bin/python
import cv2
import os
import numpy as np

def do_mosaic(img, xy, neighbor = 9):
    """
    马赛克的实现原理是把图像上某个像素点一定范围邻域内的所有点用邻域内左上像素点的颜色代替，
    这样可以模糊细节，但是可以保留大体的轮廓。
    :param img: 输入图像
    :param x: 马赛克左顶点横坐标
    :param y: 马赛克左顶点纵坐标
    :param w: 马赛克宽
    :param h: 马赛克高
    :param neighbor: 马赛克每一块的宽
    :return:
    """
    x, y, w, h = xy
    fh, fw = img.shape[0],img.shape[1]
    if(y + h > fh):
        h = fh - y
    if(x + w > fw):
        w = fw - x

    for i in range(0, h-neighbor, neighbor):
        for j in range(0, w-neighbor, neighbor):
            rect = [j+x,i+y]
            color = img[i+y][j+x].tolist()
            left_up = (rect[0], rect[1])
            right_down = (rect[0]+neighbor-1, rect[1]+neighbor-1)
            cv2.rectangle(img, left_up, right_down, color, -1)

    return img


# im = cv2.imread('QUERY_1.jpg')
# # img = do_mosaic(im, 219, 61, 460 - 219, 412 - 61)
# # cv2.imshow('mosaic',img)
# # cv2.waitKey()


root_path = 'test'   #target文件夹
#save_path = 'D://1000_Mosaic//'   #存放地址
img_list = os.listdir(root_path)

for i in range(1,11):
    for file in img_list:
        try:
            path = os.path.join(root_path, file)
            img = cv2.imread(path)
            mosaic_list = [[219, 61, 460 - 219, 412 - 61],
                           [0, 0, img.shape[0], img.shape[1]],
                           [230, 25, 128, 312],
                           [560, 200, 60, 200]]
            xy = mosaic_list[np.random.choice(len(mosaic_list))]
            dst = do_mosaic(img, xy)
        #     print(save_path+file.split('.')[0] + '_pic.jpg')
            cv2.imwrite(file.split('.')[0] + '_mosaic_'+str(i)+'.jpg',dst)
            print("已完成"+file.split('.')[0])
        except AttributeError:
            pass
        continue
