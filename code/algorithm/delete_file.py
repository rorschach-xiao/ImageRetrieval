#author : xiaoyang

import os
import re

if __name__ == '__main__':
    imglist = [os.path.join('./fortyChanged',img) for img in os.listdir('./fortyChanged') if img.endswith('.jpg') ]
    for i in range(len(imglist)):
        if(re.match(r'./fortyChanged/QUERY_\d+_\d+_pic.jpg',imglist[i])!=None):
            print("delete "+imglist[i])
            os.remove(imglist[i])
        if(re.match(r'./fortyChanged/QUERY_\d+_rotate_\d+.jpg',imglist[i])!=None):
            print("delete "+imglist[i])
            os.remove(imglist[i])
    

