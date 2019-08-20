# -*- coding: utf-8 -*-
# author: xiaoyang

import os
import h5py
import numpy as np
import argparse
import time
from extract_cnn_vgg16_keras import VGGNet
import re

ap = argparse.ArgumentParser()
ap.add_argument("-database", required = True,
	help = "Path to database which contains images to be indexed")
ap.add_argument("-index", required = True,
	help = "Name of index file")
args = vars(ap.parse_args())


'''
 Returns a list of filenames for all jpg images in a directory. 
'''
def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if (f.endswith('.jpg') or f.endswith('.jpeg'))]


'''
 Extract features and index the images
'''

if __name__ == "__main__":
    db = args["database"]
    #db = "database/Target_Changed_HuiDu_Shuiyin_2380"
    img_list = get_imlist(db)
    img_list.sort(key=lambda x:int(re.findall(r"\d+",x)[0]))
    
    print("--------------------------------------------------")
    print("         feature extraction starts")
    print("--------------------------------------------------")
    
    feats = []
    names = []
    start = time.clock()
    model = VGGNet()
    for i, img_path in enumerate(img_list):
        try:
            norm_feat = model.extract_feat(img_path)
        except OSError:
            print('读取图片失败')
        else :
            img_name = os.path.split(img_path)[1]
            feats.append(norm_feat)
            names.append(img_name)
            print("extracting feature from image No. %d , %d images in total" %((i+1), len(img_list)))

    feats = np.array(feats)
    # print(feats)
    # directory for storing extracted features
    output = args["index"]
    #output = "featureCNN.h5"
    print("--------------------------------------------------")
    print("      writing feature extraction results ...")
    print("--------------------------------------------------")


    h5f = h5py.File(output, 'w')
    h5f.create_dataset('dataset_1', data = feats)
    # h5f.create_dataset('dataset_2', data = names)
    h5f.create_dataset('dataset_2', data = np.string_(names))
    h5f.close()

    end = time.clock()
    print('feature extraction time:'+str(end-start))
