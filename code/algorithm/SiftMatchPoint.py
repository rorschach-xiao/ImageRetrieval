# author : xiaoyang
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import argparse
import csv
import re

ap = argparse.ArgumentParser()
ap.add_argument("-query", required = True,
  	help = "Name of query index file")
ap.add_argument("-database", required = True,
 	help = "Name of database index file")
args = vars(ap.parse_args())
MIN_MATCH_COUNT = 10
def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]
query_list = get_imlist(args['query'])
query_list.sort(key=lambda x:int(re.findall(r'\d+',x)[0]))
queryNames = [f for f in os.listdir(args['query']) if f.endswith('.jpg')]
queryNames.sort(key=lambda x:int(re.findall(r'\d+',x)[0]))
database_list = get_imlist(args['database'])
database_list.sort(key=lambda x:int(re.findall(r'\d+',x)[0]))
ImgNames = [f for f in os.listdir(args['database']) if f.endswith('.jpg')]
ImgNames.sort(key=lambda x :int(re.findall(r'\d+',x)[0]))
#query_list = get_imlist('database/Target')
#query_list.sort(key=lambda x:int(re.findall(r'\d+',x)[0]))
#queryNames = os.listdir('database/Target')
#queryNames.sort(key=lambda x:int(re.findall(r'\d+',x)[0]))
#database_list = get_imlist('database/base')
#database_list.sort(key=lambda x:int(re.findall(r'\d+',x)[0]))
#ImgNames = [f for f in os.listdir('database/base') if f.endswith('.jpg')]
#ImgNames.sort(key=lambda x:int(re.findall(r'\d+',x)[0]))
with open("result.csv", "w",newline='') as csvfile:
    for i in range(len(query_list)):
        img1 = cv2.imread(query_list[i],0)
        match_points_list = []
        can_be_transform = []
        for j in range(len(database_list)):
            img2 = cv2.imread(database_list[j],0)
            sift = cv2.xfeatures2d.SIFT_create()
            kp1, des1 = sift.detectAndCompute(img1,None)
            kp2, des2 = sift.detectAndCompute(img2,None)
            FLANN_INDEX_KDTREE = 0#kd树
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50)   # or pass empty dictionary
            flann = cv2.FlannBasedMatcher(index_params,search_params)
            if(len(des1)!=0 and len(des2)!=0):
                matches = flann.knnMatch(des1,des2,k=2)
            else:
                continue
            # store all the good matches as per Lowe's ratio test.
            good = []
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append(m)
            match_points_list.append(len(good))
            print("Match Point Number between "+queryNames[i]+" and "+ImgNames[j]+"is:%d"%len(good))

            if len(good)>MIN_MATCH_COUNT:
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                matchesMask = mask.ravel().tolist()
                h,w = img1.shape
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                if (type(M)!=type(None)):
                    dst = cv2.perspectiveTransform(pts,M)
                    can_be_transform.append(1)
                else:
                    can_be_transform.append(0)
                    continue
                #img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
                rect = cv2.minAreaRect(np.squeeze(dst))
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                #cv2.drawContours(img2,[box],0,(255,255,0),3)
                print('detect the area of origin picture')
                #cv2.imshow("contours", img2)
                #cv2.waitKey()
            else:
                can_be_transform.append(1)
    rank_ID = np.argsort(np.multiply(match_points_list,can_be_transform))[::-1]
    imlist = [str(ImgNames[index],encoding='utf-8') for i, index in enumerate(rank_ID[0:10])]
    writer = csv.writer(csvfile,delimiter='\t')
    writer.writerow([str(queryNames[i],encoding='utf-8')]+imlist)

