# author: xiaoyang

import h5py
import numpy as np
import argparse
import os
import re
# ap = argparse.ArgumentParser()
#
# ap.add_argument("h5index", required = True,
# 	help = "Name of database index file")
# ap.add_argument("output", required = True,
# 	help = "name of merged index file")
#
# args = vars(ap.parse_args())
#

def get_h5list(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.h5')]

if __name__ =='__main__':
    #db = args['h5index']
    db = './'
    #db = os.listdir(db)
    h5_list = get_h5list(db)
    h5_list.sort(key=lambda x:int(re.findall(r"\d+",x)[0]))
    for i in range(len(h5_list)):
        h5f_db = h5py.File(h5_list[i], 'r')
        feats_db = h5f_db['dataset_1'][:]
        imgNames = h5f_db['dataset_2'][:]
        if(i==0):
            result_feat = feats_db
            result_name = imgNames
        else:
            result_feat = np.concatenate([result_feat,feats_db],axis=0)
            result_name = np.concatenate([result_name,imgNames],axis=0)
    #output = args['output']
    output = 'feature.h5'
    h5f = h5py.File(output, 'w')
    h5f.create_dataset('dataset_1', data=result_feat)

    h5f.create_dataset('dataset_2', data=result_name)
    h5f.close()
