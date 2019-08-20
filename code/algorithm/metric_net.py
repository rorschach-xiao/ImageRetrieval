# author : xiaoyang
import numpy as np
import tensorflow as tf
class MetricNet():
    def __init__(self,input_feats):
        self.concat = input_feats
        self.fc_layer()
        self.match_score = self.fc3


    def FC(self,name,input_data,out_channel,trainable=True,flag=0):
        shape = input_data.get_shape().as_list()
        if(len(shape)==4):
            size = shape[-1]*shape[-2]*shape[-3]
        else:
            size = shape[1]
        input_data_flatten = tf.reshape(input_data,[-1,size])
        with tf.variable_scope(name):
            weights = tf.get_variable(name="weights",shape=[size,out_channel],dtype=tf.float32,trainable=trainable,
                                      initializer=tf.contrib.layers.xavier_initializer(uniform=True))
            bias = tf.get_variable(name="bias",shape=[out_channel],dtype=tf.float32,trainable=trainable,
                                   initializer=tf.constant_initializer(value=0.1,dtype=tf.float32))

            res = tf.matmul(input_data_flatten,weights)
            if(flag == 0):
                out = tf.nn.leaky_relu(tf.nn.bias_add(res,bias),alpha=0.1)
            else:
                out = tf.nn.bias_add(res,bias)
        return out


    def fc_layer(self):
        self.fc1 = self.FC("fc1",self.concat,256,trainable=True,flag=0)
        self.fc2 = self.FC("fc2",self.fc1,256,trainable=True,flag=0)
        self.fc3 = self.FC("fc3",self.fc2,2,trainable=True,flag=1)


