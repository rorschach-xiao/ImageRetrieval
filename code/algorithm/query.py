# -*- coding: utf-8 -*-
# author: xiaoyang
import tensorflow as tf
import h5py
import numpy as  np
import csv
import argparse
import re

DATA_DIR='C:/Users/A0033423/PycharmProjects/ImageRetrieval/'
MODEL_DIR='C:/Users/A0033423/PycharmProjects/ImageRetrieval/model/'

ap = argparse.ArgumentParser()
ap.add_argument("-queryindex", required = True,
	help = "Name of query index file")
ap.add_argument("-index", required = True,
	help = "Name of database index file")
ap.add_argument("-model", required = True,
	help = "Path to model file")
ap.add_argument("-MaxRes", required = True,
	help = "Max number of result")
ap.add_argument("-result", required = True,
	help = "name of the result csv file")
args = vars(ap.parse_args())


h5f_db = h5py.File(args['index'],'r')
#h5f_db = h5py.File('featureCNN.h5','r')
feats_db = h5f_db['dataset_1'][:]

TopN=60
FEAT_NUM = len(feats_db)


def eval():
    tf.reset_default_graph()
    #saver_restore = tf.train.import_meta_graph(args['model']+"MetricNet.ckpt-100.meta")
    saver_restore = tf.train.import_meta_graph(args['model'] + "MetricNet.ckpt-11.meta")
    graph = tf.get_default_graph()
    #get the parameter
    input_query_feat = graph.get_tensor_by_name("query_feat:0")
    input_db_feat = graph.get_tensor_by_name("database_feat:0")
    output_score = tf.nn.softmax(graph.get_tensor_by_name("fc3/BiasAdd:0"))

    #start the session
    with tf.Session() as sess:
        #saver_restore.restore(sess,tf.train.latest_checkpoint(args['model']))
        saver_restore.restore(sess, tf.train.latest_checkpoint(args['model']))
        h5f_db = h5py.File(args['index'], 'r')
        #h5f_db = h5py.File('featureCNN.h5', 'r')
        feats_db = h5f_db['dataset_1'][:]
        imgNames = h5f_db['dataset_2'][:]
        h5f_query = h5py.File(args['queryindex'], 'r')
        #h5f_query = h5py.File('featureCNN_Q.h5', 'r')
        feats_query = h5f_query['dataset_1'][:]
        queryNames = h5f_query['dataset_2'][:]
        with open(args['result'], "w",newline='') as csvfile:
            recall_list =[]
            for i in range(len(feats_query)):
                correct_retrieval_num = 0
                one_query_feat = np.reshape(feats_query[i],(1,512))
                temp_query_feat_matrix = one_query_feat.repeat(FEAT_NUM,axis=0)
                scores = sess.run(output_score,feed_dict={input_query_feat:temp_query_feat_matrix,input_db_feat:feats_db})
                scores_fc = np.array(scores)[:,1]
                scores_cos = np.dot(feats_query[i],feats_db.T)
                final_scores = np.multiply(scores_fc,scores_cos)
                rank_ID = np.argsort(final_scores)[::-1]
                rank_name = imgNames[rank_ID]
                #print(rank_score)
                maxres = int(args['MaxRes'])
                imlist = [str(imgNames[index],encoding='utf-8') for i, index in enumerate(rank_ID[0:maxres])]
                for j in range(len(imlist)): # calculate the correct recall number
                    if(rank_ID[j] in range(i*maxres,i*maxres+maxres)):
                        correct_retrieval_num+=1
                recall_i = correct_retrieval_num/maxres
                recall_list.append(recall_i)
                print("Query No.%d Recall:"%i+str(recall_i))
                writer = csv.writer(csvfile,delimiter = '\t')
                writer.writerow([str(queryNames[i],encoding='utf-8')]+imlist)
            total_recall = np.mean(recall_list)
            print("Total Recall:"+str(total_recall))

if __name__ =='__main__':
    eval()





