import tensorflow as tf
import cv2
import numpy as np

USE_BN = 1
BATCH_SIZE = 128
WEIGHT_DECAY = 0.01
INIT_LR = 0.001

def _conv(name,input,kernel_size,in_filters,out_filters,strides):
    with tf.variable_scope(name):
        filter = tf.get_variable('DW',[kernel_size,kernel_size,in_filters,out_filters],tf.float32,
                                 tf.truncated_normal_initializer(stddev=0.1))
        return tf.nn.conv2d(input,filter,[1,strides,strides,1],padding = 'SAME')

def _leaky_relu(x, leakiness=0.1):
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

def _FC(name,input,out_dim,keep_rate,activation='relu'):
    assert (activation == 'relu') or (activation == 'softmax') or (activation == 'linear')
    with tf.variable_scope(name):
        dim = input.get_shape().as_list()
        dim = np.prod(dim[1:]) #计算所有元素的乘积
        x = tf.reshape(input, [-1, dim])
        W = tf.get_variable('DW', [x.get_shape()[1], out_dim],
                            initializer=tf.random_normal_initializer(stddev=0.1))
        b = tf.get_variable('bias', [out_dim], initializer=tf.constant_initializer())
        x = tf.nn.xw_plus_b(x, W, b)
        if activation == 'relu':
            x = _leaky_relu(x)
        else:
            if activation == 'softmax':
                x = tf.nn.softmax(x)

        if activation != 'relu':
            return x
        else:
            return tf.nn.dropout(x, keep_rate)
def _maxpooling(input,kernel_size,strides):
    return tf.nn.max_pool(input,[1,kernel_size,kernel_size,1],[1,strides,strides,1])

def _batchnorm(input,n_out,phase_train,scope = 'bn'):
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0,shape=[n_out]),
                           name='beta',trainable=True)
        gamma = tf.Variable(tf.constant(1.0),shape=[n_out],
                            name='gamma',trainable=True)
        batch_mean ,batch_var = tf.nn.moments(input,[0,1,2],name='moments')
        ema = tf.train.ExponentialMovingAverage(decay = 0.5)
        def mean_var_with_updata():
            ema_apply_op = ema.apply([batch_mean,batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean),tf.identity(batch_var)
        mean,var = tf.cond(phase_train,
                           mean_var_with_updata,
                           lambda:(ema.average(batch_mean),ema.apply(batch_var)))
        normed = tf.nn.batch_normalization(input,mean,var,beta,gamma,1e-3)
    return normed

def VGG_Conv_Block(name,input,in_filters,out_filters,repeat,strides,phase_train):
    with tf.variable_scope(name):
        for layer in repeat:
            scope_name = name+'_'+str(layer)
            x = _conv(scope_name,input,3,in_filters,out_filters,strides)
            if USE_BN:
                x = _batchnorm(x,out_filters,phase_train)
            x = _leaky_relu(x)
            in_filters = out_filters
        x = _maxpooling(x,2,2)
        return x
def Input_Img():
    x = tf.placeholder(tf.float32,[None,None,None,3])
    y = tf.placeholder(tf.float32,[None])

    return x,y
def LandmarkNet(x):
    phase_train = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32)

    x = VGG_Conv_Block('BLOCK1',x,3,32,1,1,phase_train)

    x = VGG_Conv_Block('BLOCK2',x,32,64,1,1,phase_train)

    x = VGG_Conv_Block('BLOCK3',x,64,64,1,1,phase_train)

    x = _conv('BLOCK4_conv1',x,2,64,128,1)

    landmark_fc1 = _FC('fc1',x,256,keep_prob)
    landmark_fc2 = _FC('fc2',landmark_fc1,256,keep_prob)
    y_landmark_conv = _FC('fc3',landmark_fc2,4,keep_prob)

    return y_landmark_conv,phase_train,keep_prob

def landmark_loss(y_,y_landmark_conv):
    with tf.variable_scope('loss'):
        y_landmark = tf.slice(y_, [0, 0], [BATCH_SIZE, 4])
        tf.add_to_collection('y_ethnic', y_landmark)
        smooth_l1_loss = tf.losses.huber_loss(y_landmark,y_landmark_conv)
        l2_loss = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'DW') > 0:
                l2_loss.append(tf.nn.l2_loss(var))
        l2_loss = WEIGHT_DECAY * tf.add_n(l2_loss)

        total_loss = smooth_l1_loss+l2_loss
        return smooth_l1_loss,l2_loss,total_loss

def train_op(loss,global_step):
    learning_rate = tf.train.exponential_decay(INIT_LR,global_step,decay_steps=2000,decay_rate=0.95,staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    tf.add_to_collection('learning_rate',learning_rate)
    return optimizer









