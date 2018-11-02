#coding:utf-8
import tensorflow as tf
#import tensorflow.contrib.slim as slim
from tensorflow.contrib import slim
import numpy as np
num_keep_radio = 0.7
#define prelu
def prelu(inputs):
    alphas = tf.get_variable("alphas", shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(0.25))
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs-abs(inputs))*0.5
    return pos + neg
def dense_to_one_hot(labels_dense,num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels)*num_classes
    #num_sample*num_classes
    labels_one_hot = np.zeros((num_labels,num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
#cls_prob:batch*2
#label:batch

def cls_ohem(cls_prob, label):

    zeros = tf.zeros_like(label)
    #label=-1 --> label=0net_factory
    
    label_filter_invalid = tf.where(tf.less(label,0), zeros, label)  #
    '''
    ones = tf.ones_like(label)
    label_filter_invalid = tf.where(label != zeros, ones, label)  #
    '''
    num_cls_prob = tf.size(cls_prob)
    cls_prob_reshape = tf.reshape(cls_prob,[num_cls_prob,-1])
    label_int = tf.cast(label_filter_invalid,tf.int32)
    num_row = tf.to_int32(cls_prob.get_shape()[0])
    row = tf.range(num_row)*2
    indices_ = row + label_int  #the pos indices add 1
    label_prob = tf.squeeze(tf.gather(cls_prob_reshape, indices_))
    loss = -tf.log(label_prob+1e-10)
    zeros = tf.zeros_like(label_prob, dtype=tf.float32)
    ones = tf.ones_like(label_prob,dtype=tf.float32)

    valid_inds = tf.where(label < zeros,zeros,ones)  #
    num_valid = tf.reduce_sum(valid_inds)  #not compute part and landmark

    keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    #set 0 to invalid sample
    loss = loss * valid_inds
    loss,_ = tf.nn.top_k(loss, k=keep_num)
    return tf.reduce_mean(loss)
def bbox_ohem_smooth_L1_loss(bbox_pred,bbox_target,label):
    sigma = tf.constant(1.0)
    threshold = 1.0/(sigma**2)
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    valid_inds = tf.where(label!=zeros_index,tf.ones_like(label,dtype=tf.float32),zeros_index)
    abs_error = tf.abs(bbox_pred-bbox_target)
    loss_smaller = 0.5*((abs_error*sigma)**2)
    loss_larger = abs_error-0.5/(sigma**2)
    smooth_loss = tf.reduce_sum(tf.where(abs_error<threshold,loss_smaller,loss_larger),axis=1)
    keep_num = tf.cast(tf.reduce_sum(valid_inds)*num_keep_radio,dtype=tf.int32)
    smooth_loss = smooth_loss*valid_inds
    _, k_index = tf.nn.top_k(smooth_loss, k=keep_num)
    smooth_loss_picked = tf.gather(smooth_loss, k_index)
    return tf.reduce_mean(smooth_loss_picked)
def bbox_ohem_orginal(bbox_pred,bbox_target,label):
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    #pay attention :there is a bug!!!!
    valid_inds = tf.where(label!=zeros_index,tf.ones_like(label,dtype=tf.float32),zeros_index)
    #(batch,)
    square_error = tf.reduce_sum(tf.square(bbox_pred-bbox_target),axis=1)
    #keep_num scalar
    keep_num = tf.cast(tf.reduce_sum(valid_inds)*num_keep_radio,dtype=tf.int32)
    #keep valid index square_error
    square_error = square_error*valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)
#label=1 or label=-1 then do regression
def bbox_ohem(bbox_pred,bbox_target,label):
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    ones_index = tf.ones_like(label,dtype=tf.float32)
    valid_inds = tf.where(tf.equal(tf.abs(label), 1),ones_index,zeros_index)
    #(batch,)
    square_error = tf.square(bbox_pred-bbox_target)
    square_error = tf.reduce_sum(square_error,axis=1)
    #keep_num scalar
    num_valid = tf.reduce_sum(valid_inds)
    #keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    #keep valid index square_error
    square_error = square_error*valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)

def landmark_ohem(landmark_pred,landmark_target,label):
    #keep label =-2  then do landmark detection
    ones = tf.ones_like(label,dtype=tf.float32)
    zeros = tf.zeros_like(label,dtype=tf.float32)
    valid_inds = tf.where(tf.equal(label,-2),ones,zeros)
    square_error = tf.square(landmark_pred-landmark_target)
    square_error = tf.reduce_sum(square_error,axis=1)
    num_valid = tf.reduce_sum(valid_inds)
    #keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    square_error = square_error*valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)
    
def cal_accuracy(cls_prob,label):
    pred = tf.argmax(cls_prob,axis=1)
    label_int = tf.cast(label,tf.int64)
    cond = tf.where(tf.greater_equal(label_int,0))
    picked = tf.squeeze(cond)
    label_picked = tf.gather(label_int,picked)
    pred_picked = tf.gather(pred,picked)
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(label_picked,pred_picked),tf.float32))
    return accuracy_op
#construct Pnet
#label:batch

    
def O_Net(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn = prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),                        
                        padding='valid'):
        print ("inputs",inputs.get_shape())
        net = slim.conv2d(inputs, num_outputs=32, kernel_size=[3,3], stride=1, scope="conv1")
        print (net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool1", padding='SAME')
        print (net.get_shape())
        net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope="conv2")
        print (net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool2")
        print (net.get_shape())
        net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope="conv3")
        print (net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope="pool3", padding='SAME')
        print (net.get_shape())
        net = slim.conv2d(net,num_outputs=128,kernel_size=[2,2],stride=1,scope="conv4")
        print (net.get_shape())               
        fc_flatten = slim.flatten(net)
        print (fc_flatten.get_shape())
        fc1 = slim.fully_connected(fc_flatten, num_outputs=256,scope="fc1", activation_fn=prelu)
        print (fc1.get_shape())
        #batch*2
        cls_prob = slim.fully_connected(fc1,num_outputs=2,scope="cls_fc",activation_fn=tf.nn.softmax)
        print (cls_prob.get_shape())
        #batch*4
        bbox_pred = slim.fully_connected(fc1,num_outputs=4,scope="bbox_fc",activation_fn=None)
        print (bbox_pred.get_shape())
        #batch*10
        landmark_pred = slim.fully_connected(fc1,num_outputs=10,scope="landmark_fc",activation_fn=None)
        print (landmark_pred.get_shape())        
        #train
        if training:
            cls_loss = cls_ohem(cls_prob,label)
            bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
            accuracy = cal_accuracy(cls_prob,label)
            landmark_loss = landmark_ohem(landmark_pred, landmark_target,label)
            L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
        else:
            return cls_prob,bbox_pred,landmark_pred

def O_Net_new(inputs,label=None,bbox_target=None,landmark_target=None,training=True):  
    #define common param
    #is_training=training
    #dropout_keep_prob=0.8
    #bottleneck_layer_size=1000,
    width_multiplier=1
    weight_decay=0.0005
    #reuse = None,
    
    batch_norm_params = {
    # Decay for the moving averages.
    'decay': 0.995,
    # epsilon to prevent 0s in variance.
    'epsilon': 0.001,
    #scale
    #'scale': True,
    # force in-place updates of mean and variance estimates
    'updates_collections': None,
    # Moving averages ends up in the trainable variables collection
    'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
    }
    
    def _depthwise_separable_conv(inputs,
                                num_pwc_filters,
                                width_multiplier,
                                sc,
                                downsample=False):
        """ Helper function to build the depth-wise separable convolution layer.
        """
        #num_pwc_filters = #round(num_pwc_filters * width_multiplier)
        _stride = 2 if downsample else 1

        # skip pointwise by setting num_outputs=None
        depthwise_conv = slim.separable_convolution2d(inputs,
                                                    num_outputs=None,
                                                    stride=_stride,
                                                    depth_multiplier=1,
                                                    kernel_size=[3, 3],
                                                    scope=sc + '/depthwise_conv',
                                                    padding='VALID')
        #print(depthwise_conv.get_shape())
        #bn = slim.batch_norm(depthwise_conv, scope=sc + '/dw_batch_norm')

        pointwise_conv = slim.convolution2d(depthwise_conv,
                                            num_pwc_filters,
                                            kernel_size=[1, 1],
                                            scope=sc + '/pointwise_conv',
                                            padding='SAME')
        #print(pointwise_conv.get_shape())
        #bn = slim.batch_norm(pointwise_conv, scope=sc + '/pw_batch_norm')
        return pointwise_conv

    
    with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                        weights_initializer=slim.xavier_initializer_conv2d(),
                        biases_initializer=slim.init_ops.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay),#l1_regularizer?
                        normalizer_fn = slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        #activation_fn = prelu,
                        #outputs_collections=[end_points_collection],
                        #):
                        padding = 'VALID'):
        
        """net = slim.convolution2d(inputs, 16, [3, 3], stride=2, padding='SAME', scope='conv_1')#32==>16
        net = _depthwise_separable_conv(net, 24, 1, downsample=True, sc='conv_ds_2')#16==>8
        net = _depthwise_separable_conv(net, 32, 1, downsample=True, sc='conv_ds_3')#8==>4
        net = _depthwise_separable_conv(net, 64, 1, downsample=True, sc='conv_ds_4')#4==>2
        net = slim.avg_pool2d(net, [2, 2], scope='avg_pool_5')"""
        print(inputs.get_shape())        
        net = slim.convolution2d(inputs, round(32 * width_multiplier), [3, 3], stride=1, padding='valid', scope='conv_1')
        #net = _depthwise_separable_conv(inputs, 32,  width_multiplier, sc='conv_ds_1')
        print(net.get_shape())

        net = _depthwise_separable_conv(net, 32,  width_multiplier, downsample=True,sc='conv_ds_2')
        print(net.get_shape())
      
        net = _depthwise_separable_conv(net, 64, width_multiplier, sc='conv_ds_4')
        print(net.get_shape())
        net = _depthwise_separable_conv(net, 64, width_multiplier, downsample=True, sc='conv_ds_3')
        
        print(net.get_shape())
        net = _depthwise_separable_conv(net, 64, width_multiplier, sc='conv_ds_41')
        print(net.get_shape())
        #net = _depthwise_separable_conv(net, 64, width_multiplier, sc='conv_ds_4')
        #print(net.get_shape())

        #net = _depthwise_separable_conv(net, 256, width_multiplier, downsample=True, sc='conv_ds_4')

        net = _depthwise_separable_conv(net, 128, width_multiplier, downsample=True, sc='conv_ds_5')
        print(net.get_shape())

    
        fc_flatten = slim.flatten(net)
        print (fc_flatten.get_shape())
        fc1 = slim.fully_connected(fc_flatten, num_outputs=256,scope="fc1", activation_fn=prelu)
        print (fc1.get_shape())


        #cls_prob = slim.conv2d(net,num_outputs=2,kernel_size=[1,1],stride=1,scope='conv4_1',activation_fn=tf.nn.softmax)
        cls_prob =  slim.fully_connected(fc1, 2, activation_fn=tf.nn.softmax, scope='cls_fc') 
        print (cls_prob.get_shape())
  
        #train
        if training:
            cls_loss = cls_ohem(cls_prob,label)
            #bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
            accuracy = cal_accuracy(cls_prob,label)
            #print('landmark_target.get_shape()',landmark_target.get_shape())
            #landmark_loss = landmark_ohem(landmark_pred, landmark_target,label)
            L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            return cls_loss,L2_loss,accuracy
        else:
            return cls_prob

def O_Net_new1(inputs,label=None,bbox_target=None,landmark_target=None,training=True):  
    #define common param
    #is_training=training
    #dropout_keep_prob=0.8
    #bottleneck_layer_size=1000,
    width_multiplier=1
    weight_decay=0.0005
    #reuse = None,
    
    batch_norm_params = {
    # Decay for the moving averages.
    'decay': 0.995,
    # epsilon to prevent 0s in variance.
    'epsilon': 0.001,
    #scale
    #'scale': True,
    # force in-place updates of mean and variance estimates
    'updates_collections': None,
    # Moving averages ends up in the trainable variables collection
    'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
    }
    
    def _depthwise_separable_conv(inputs,
                                num_pwc_filters,
                                width_multiplier,
                                sc,
                                downsample=False):
        """ Helper function to build the depth-wise separable convolution layer.
        """
        num_pwc_filters = round(num_pwc_filters * width_multiplier)
        _stride = 2 if downsample else 1

        # skip pointwise by setting num_outputs=None
        depthwise_conv = slim.separable_convolution2d(inputs,
                                                    num_outputs=None,
                                                    stride=_stride,
                                                    depth_multiplier=1,
                                                    kernel_size=[3, 3],
                                                    scope=sc + '/depthwise_conv')
                                                    #padding='VALID')
        #print(depthwise_conv.get_shape())
        #bn = slim.batch_norm(depthwise_conv, scope=sc + '/dw_batch_norm')

        pointwise_conv = slim.convolution2d(depthwise_conv,
                                            num_pwc_filters,
                                            kernel_size=[1, 1],
                                            scope=sc + '/pointwise_conv',
                                            padding='SAME')
        #print(pointwise_conv.get_shape())
        #bn = slim.batch_norm(pointwise_conv, scope=sc + '/pw_batch_norm')
        return pointwise_conv

    
    with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                        weights_initializer=slim.xavier_initializer_conv2d(),
                        biases_initializer=slim.init_ops.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay),#l1_regularizer?
                        normalizer_fn = slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        #activation_fn = prelu,
                        #outputs_collections=[end_points_collection],
                        ):
                        #padding = 'VALID'):
        with slim.arg_scope([slim.batch_norm],
                              is_training=training,
                             ):
 
            print(inputs.get_shape())        
            net = slim.convolution2d(inputs, round(16 * width_multiplier), [3, 3], stride=1, padding='SAME', scope='conv_1')
            #net = _depthwise_separable_conv(inputs, 32,  width_multiplier, sc='conv_ds_1')
            print(net.get_shape())
            
            net = _depthwise_separable_conv(net, 32,  width_multiplier, downsample=True,sc='conv_ds_2')
            print(net.get_shape())
        
            net = _depthwise_separable_conv(net, 64, width_multiplier,downsample=True, sc='conv_ds_3')
            print(net.get_shape())
            net = _depthwise_separable_conv(net, 64, width_multiplier, downsample=True, sc='conv_ds_4')
            
            print(net.get_shape())
            net = _depthwise_separable_conv(net, 128, width_multiplier, downsample=True, sc='conv_ds_5')
            print(net.get_shape())
            #net = _depthwise_separable_conv(net, 64, width_multiplier, sc='conv_ds_4')
            #print(net.get_shape())
            net = slim.avg_pool2d(net, [3, 3], scope='avg_pool_15')
            print(net.get_shape())
            #net = _depthwise_separable_conv(net, 256, width_multiplier, downsample=True, sc='conv_ds_6')
            #print(net.get_shape())

            fc_flatten = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
            print(fc_flatten.get_shape())
        
            #fc_flatten = slim.flatten(net)
            #print (fc_flatten.get_shape())
            fc1 = slim.fully_connected(fc_flatten, num_outputs=256*width_multiplier,scope="fc1")
            print (fc1.get_shape())


            #cls_prob = slim.conv2d(net,num_outputs=2,kernel_size=[1,1],stride=1,scope='conv4_1',activation_fn=tf.nn.softmax)
            cls_prob =  slim.fully_connected(fc1, 2, activation_fn=tf.nn.softmax, scope='cls_fc') 
            print (cls_prob.get_shape())

        #train
        if training:
            cls_loss = cls_ohem(cls_prob,label)
            accuracy = cal_accuracy(cls_prob,label)
            L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            return cls_loss ,L2_loss ,accuracy
        else:
            return cls_prob #,bbox_pred,landmark_pred







        
        
                                                                  
