# author: xiaoyang
import numpy as np
import h5py
from datetime import date
import time
import random
import argparse

import tensorflow as tf
import metric_net

#DATA_DIR='C:/Users/A0033423/PycharmProjects/ImageRetrieval/'
#LOGS_DIR='C:/Users/A0033423/PycharmProjects/ImageRetrieval/logs'
#RUNNING_LOG_DIR='C:/Users/A0033423/PycharmProjects/ImageRetrieval/running_logs'
#MODEL_DIR='C:/Users/A0033423/PycharmProjects/ImageRetrieval/model'
RUNNING_LOG_DIR='running_logs'
LOGS_DIR='logs'

ap = argparse.ArgumentParser()
ap.add_argument("-queryindex", required = True,
  	help = "Name of query index file")
ap.add_argument("-index", required = True,
    help = "Name of database index file")
ap.add_argument("-negativeindex", required = True,
    help = "Name of database index file")
ap.add_argument("-model", required = True,
  	help = "Path to model file")
ap.add_argument("-MaxRes", required = True,
    help = "Path to model file")
ap.add_argument("-trainmode", required = True,
    help = "select train mode")
args = vars(ap.parse_args())

#h5f_query = h5py.File(DATA_DIR + 'featureCNN_Q_big.h5', 'r')
h5f_query = h5py.File(args['queryindex'], 'r')
feats_query = h5f_query['dataset_1'][:]


LEARNING_RATE = 1e-4
TRAIN_ROUND = 300
FEAT_NUM = len(feats_query)
TRAIN_NUM = int(np.ceil(FEAT_NUM*0.8))
TEST_NUM = FEAT_NUM-TRAIN_NUM
maxres = int(args['MaxRes'])
BATCH_SIZE = 2*maxres
FLAG = int(args['trainmode'])


def train():
    norm_feat_query = tf.placeholder(tf.float32,[None,512],name = 'query_feat')
    norm_feat_db = tf.placeholder(tf.float32,[None,512],name = 'database_feat')
    concat_feat = tf.concat((norm_feat_query,norm_feat_db),axis=1,name='input_concat')
    y = tf.placeholder(tf.float32,[None,2],name = 'ground_truth_score')
    model = metric_net.MetricNet(concat_feat)
    match_score = model.match_score

    #define loss and optimizer
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=match_score)
    #loss = tf.nn.weighted_cross_entropy_with_logits(logits=match_score,labels=y,pos_weight=tf.constant(0.5))
    #loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=match_score)
    mean_loss = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(mean_loss)
    #optimizer = tf.train.AdadeltaOptimizer(learning_rate=LEARNING_RATE).minimize(mean_loss)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(mean_loss)
    tf.summary.scalar('mean_loss',mean_loss)

    #create the session and initializer the parameters
    sess=tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOGS_DIR, sess.graph)
    saver = tf.train.Saver(max_to_keep=1) #
    max_acc = 0.8

    #preprocessing data
    #h5f_db = h5py.File(DATA_DIR + 'featureCNN_big.h5', 'r')
    h5f_db = h5py.File(args['index'], 'r')
    feats_db = h5f_db['dataset_1'][:]
    #h5f_query = h5py.File(DATA_DIR + 'featureCNN_Q_big.h5', 'r')
    h5f_query = h5py.File(args['queryindex'], 'r')
    h5f_db_neg = h5py.File(args['negativeindex'], 'r')
    #h5f_db_neg = h5py.File(DATA_DIR+'featureCNN_neg.h5', 'r')
    feats_db_neg = h5f_db_neg['dataset_1'][:]
    feats_query = h5f_query['dataset_1'][:]

    train_feats_query = np.array(feats_query[0:TRAIN_NUM])
    train_feats_db = feats_db[0:TRAIN_NUM*maxres]
    train_feats_neg_other = feats_db_neg[0:24000]

    test_feats_query = np.array(feats_query[TRAIN_NUM:])
    test_feats_db = feats_db[TRAIN_NUM*maxres:]
    test_feats_neg_other = feats_db_neg[24000:]
    test_positive = np.concatenate((test_feats_query.repeat(maxres,axis=0),test_feats_db),axis=1)
    for i in range(TEST_NUM): 
        test_feats_db_neg_one = random.sample(list(test_feats_db[0:i*maxres])+list(test_feats_db[i*maxres+maxres:]),maxres)
        #test_feats_db_neg_one = random.sample(list(test_feats_neg_other), maxres)
        test_feats_query_neg_one = np.array(np.reshape(test_feats_query[i],(1,512))).repeat(maxres,axis=0)
        test_feats_neg = np.concatenate([test_feats_query_neg_one,test_feats_db_neg_one],axis = 1)
        if (i==0):
            test_negative = test_feats_neg
        else:
            test_negative= np.concatenate([test_negative,test_feats_neg],axis=0)


    #start training
    with open(RUNNING_LOG_DIR+"/log"+date.isoformat(date.today())+str(time.time())+".txt","w+") as file:
        file.write('BATCH SIEZ:' + str(BATCH_SIZE) + '\n' + ' TRAINING ROUND:')
        last_test_loss_list = []
        for epoch in range(TRAIN_ROUND):
            test_loss_list = []
            train_loss_list = []
            acc_list = []
            for i in range(0,TRAIN_NUM):
                positive_sample_num = 0
                # # random select negative sample
                train_feats_db_neg = random.sample(list(train_feats_db[0:i*maxres]) + list(train_feats_db[i*maxres+maxres:]),
                                                        maxres)  
                #train_feats_db_neg = random.sample(list(train_feats_neg_other), maxres)
                train_feats_db_pos = train_feats_db[i*maxres:i*maxres+maxres]  
                train_data_query = np.reshape(train_feats_query[i],(1,512)).repeat(maxres*2,axis=0) 
                train_data_db = np.concatenate([train_feats_db_pos,train_feats_db_neg],axis = 0)
                train_data = np.concatenate([train_data_query,train_data_db],axis=1)
                label = np.concatenate((np.tile([0,1],[maxres,1]),np.tile([1,0],[maxres,1])),axis=0)
                label_test = np.concatenate((np.tile([0,1],[50,1]),np.tile([1,0],[50,1])),axis=0)

                # random select test sample
                test_data_pos = random.sample(list(test_positive), 50)   
                test_data_neg = random.sample(list(test_negative), 50)  
                test_data = np.concatenate([test_data_pos, test_data_neg], axis=0)
                # get train loss
                result ,optimizer_res,train_loss = sess.run([merged,optimizer,mean_loss],feed_dict={concat_feat:train_data,y:label})
                # get test loss
                test_res, test_loss = sess.run([match_score, mean_loss], feed_dict={concat_feat: test_data,y:label_test})
                test_loss_list.append(test_loss)
                train_loss_list.append(train_loss)

                # caculate test accuracy
                for j in range(len(test_res)):
                    if((test_res[j][0]<test_res[j][1] and j<50) or (test_res[j][0]>test_res[j][1] and j>=50)):
                        positive_sample_num +=1
                test_accuracy = positive_sample_num/100
                acc_list.append(test_accuracy)
                #show train process
                if(i==0):
                    print("------------Epoch%d-------------"%epoch)
                    file.write('round ' + str(epoch) + ' batch ' + str(i) + ' train_loss ' + str(train_loss) + '\n')
                print("step %d"%(i)+" train loss:"+str(train_loss)+" test loss:"+
                      str(test_loss)+" test acc:"+str(test_accuracy))
                writer.add_summary(result, epoch)
            # save model
            mean_acc = np.mean(acc_list)
            print("----------------------current accuracy:%f-------------------------" %mean_acc)
            if(mean_acc>max_acc):
                max_acc = mean_acc
                saver.save(sess, args['model']+'/MetricNet.ckpt', global_step=epoch + 1)  
            # early stopping
            if (len(last_test_loss_list) != 0):
                train_mean_loss = np.mean(train_loss_list)
                test_mean_loss = np.mean(test_loss_list)
                last_test_mean_loss = np.mean(last_test_loss_list)
                print("train_mean_loss:"+str(train_mean_loss)+"  test_mean_loss:"+str(test_mean_loss)+" last_epoch_test_mean_loss:"+str(last_test_mean_loss))
                if (test_mean_loss> last_test_mean_loss and train_mean_loss < test_mean_loss and epoch>5):
                    print("=================about to overfitting===================")
                    exit()
            last_test_loss_list = test_loss_list
def retrain():
    tf.reset_default_graph()
    saver_restore = tf.train.import_meta_graph(args['model']+"/MetricNet.ckpt-11.meta")
    graph = tf.get_default_graph()

    input_norm_feat_db = graph.get_tensor_by_name("database_feat:0")
    input_norm_feat_query = graph.get_tensor_by_name("query_feat:0")
    input_concat_feat = graph.get_tensor_by_name("input_concat:0")
    final_loss = graph.get_tensor_by_name("Mean:0")
    gt_score = graph.get_tensor_by_name("ground_truth_score:0")
    match_score = graph.get_tensor_by_name("fc3/BiasAdd:0")

    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE,name='Adam01').minimize(final_loss)
    tf.summary.scalar('mean_loss', final_loss)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOGS_DIR, sess.graph)
    saver = tf.train.Saver(max_to_keep=1)  
    saver_restore.restore(sess,tf.train.latest_checkpoint(args['model']))
    max_acc = 0.8

    # preprocessing data
    #h5f_db = h5py.File(DATA_DIR + 'featureCNN_big.h5', 'r')
    h5f_db = h5py.File(args['index'], 'r')
    feats_db = h5f_db['dataset_1'][:]
    #h5f_query = h5py.File(DATA_DIR + 'featureCNN_Q_big.h5', 'r')
    h5f_query = h5py.File(args['queryindex'], 'r')
    h5f_db_neg = h5py.File(args['negativeindex'], 'r')
    #h5f_db_neg = h5py.File(DATA_DIR+'featureCNN_neg.h5', 'r')
    feats_db_neg = h5f_db_neg['dataset_1'][:]
    feats_query = h5f_query['dataset_1'][:]

    train_feats_query = np.array(feats_query[0:TRAIN_NUM])
    train_feats_db = feats_db[0:TRAIN_NUM * maxres]
    train_feats_neg_other = feats_db_neg[0:24000]

    test_feats_query = np.array(feats_query[TRAIN_NUM:])
    test_feats_db = feats_db[TRAIN_NUM * maxres:]
    test_feats_neg_other = feats_db_neg[24000:]
    test_positive = np.concatenate((test_feats_query.repeat(maxres, axis=0), test_feats_db), axis=1)
    for i in range(TEST_NUM):  
        test_feats_db_neg_one = random.sample(list(test_feats_db[0:i*maxres])+list(test_feats_db[i*maxres+maxres:]),maxres)
        test_feats_db_neg_one = random.sample(list(test_feats_neg_other), maxres)
        test_feats_query_neg_one = np.array(np.reshape(test_feats_query[i], (1, 512))).repeat(maxres, axis=0)
        test_feats_neg = np.concatenate([test_feats_query_neg_one, test_feats_db_neg_one], axis=1)
        if (i == 0):
            test_negative = test_feats_neg
        else:
            test_negative = np.concatenate([test_negative, test_feats_neg], axis=0)

    # start training
    with open(RUNNING_LOG_DIR + "/log" + date.isoformat(date.today()) + str(time.time()) + ".txt", "w+") as file:
        file.write('BATCH SIEZ:' + str(BATCH_SIZE) + '\n' + ' TRAINING ROUND:')
        last_test_loss_list = []
        for epoch in range(TRAIN_ROUND):
            test_loss_list = []
            train_loss_list = []
            acc_list = []
            for i in range(0, TRAIN_NUM):
                positive_sample_num = 0
                # # random select negative sample
                train_feats_db_neg = random.sample(list(train_feats_db[0:i*maxres]) + list(train_feats_db[i*maxres+maxres:]),
                                                        int(maxres))  
                #train_feats_db_neg = random.sample(list(train_feats_neg_other), int(maxres))
                train_feats_db_pos = train_feats_db[i * maxres:i * maxres + maxres]  
                train_data_query = np.reshape(train_feats_query[i], (1, 512)).repeat(maxres * 2, axis=0)  
                train_data_db = np.concatenate([train_feats_db_pos, train_feats_db_neg], axis=0)
                train_data = np.concatenate([train_data_query, train_data_db], axis=1)
                label = np.concatenate((np.tile([0, 1], [maxres, 1]), np.tile([1, 0], [maxres, 1])), axis=0)
                label_test = np.concatenate((np.tile([0, 1], [50, 1]), np.tile([1, 0], [50, 1])), axis=0)

                # random select test sample
                test_data_pos = random.sample(list(test_positive), 50)  
                test_data_neg = random.sample(list(test_negative), 50)  
                test_data = np.concatenate([test_data_pos, test_data_neg], axis=0)
                # get train loss
                result, optimizer_res, train_loss = sess.run([merged, optimizer, final_loss],
                                                             feed_dict={input_concat_feat: train_data, gt_score: label})
                # get test loss
                test_res, test_loss = sess.run([match_score, final_loss],
                                               feed_dict={input_concat_feat: test_data, gt_score: label_test})
                test_loss_list.append(test_loss)
                train_loss_list.append(train_loss)

                # caculate test accuracy
                for j in range(len(test_res)):
                    if ((test_res[j][0] < test_res[j][1] and j < 50) or (test_res[j][0] > test_res[j][1] and j >= 50)):
                        positive_sample_num += 1
                test_accuracy = positive_sample_num / 100
                acc_list.append(test_accuracy)
                # show train process
                if (i == 0):
                    print("------------Epoch%d-------------" % epoch)
                    file.write('round ' + str(epoch) + ' batch ' + str(i) + ' train_loss ' + str(train_loss) + '\n')
                print("step %d" % (i) + " train loss:" + str(train_loss) + " test loss:" +
                      str(test_loss) + " test acc:" + str(test_accuracy))
                writer.add_summary(result, epoch)
            # save model
            mean_acc = np.mean(acc_list)
            print("----------------------current accuracy:%f-------------------------" %mean_acc)
            if (mean_acc > max_acc):
                max_acc = mean_acc
                saver_restore.save(sess, args['model'] + '/MetricNet.ckpt', global_step=epoch + 1,write_meta_graph = False)  
            # early stopping
            if (len(last_test_loss_list) != 0):
                train_mean_loss = np.mean(train_loss_list)
                test_mean_loss = np.mean(test_loss_list)
                last_test_mean_loss = np.mean(last_test_loss_list)
                print("train_mean_loss:" + str(train_mean_loss) + "  test_mean_loss:" + str(
                    test_mean_loss) + " last_epoch_test_mean_loss:" + str(last_test_mean_loss))
                if (test_mean_loss > last_test_mean_loss and train_mean_loss < test_mean_loss):
                    print("=================about to overfitting===================")
                    exit()
            last_test_loss_list = test_loss_list





if __name__ == '__main__':
    if(FLAG ==1):
        train()
    else:
        retrain()


































