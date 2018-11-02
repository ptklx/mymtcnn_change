#coding:utf-8
import tensorflow as tf
import numpy as np
import os
from datetime import datetime
import sys
sys.path.append("./prepare_data")
sys.path.append('X:\\deeplearn\\mtcnn\\MTCNN-Tensorflow-masterP\\prepare_data')
print (sys.path)
from read_tfrecord_v2 import read_multi_tfrecords,read_single_tfrecord
from MTCNN_config import config
#from mtcnn_model import P_Net
import random
import numpy.random as npr
import cv2
#import pdb
def train_model(base_lr, loss, data_num):
    """
    train model
    :param base_lr: base learning rate
    :param loss: loss
    :param data_num:
    :return:
    train_op, lr_op
    """

    lr_factor = 0.1
    global_step = tf.Variable(0, trainable=False)

    #LR_EPOCH [8,14]
    #boundaried [num_batch,num_batch]
    boundaries = [int(epoch * data_num / config.BATCH_SIZE) for epoch in config.LR_EPOCH]
    #lr_values[0.01,0.001,0.0001,0.00001]
    lr_values = [base_lr * (lr_factor ** x) for x in range(0, len(config.LR_EPOCH) + 1)]
    print(lr_values)
    #control learning rate
    lr_op = tf.train.piecewise_constant(global_step, boundaries, lr_values)
    optimizer = tf.train.MomentumOptimizer(lr_op, 0.9)
    #optimizer = tf.train.AdamOptimizer(lr_op, 0.9)
    train_op = optimizer.minimize(loss, global_step)

    return train_op, lr_op, global_step
'''
certain samples mirror
def random_flip_images(image_batch,label_batch,landmark_batch):
    num_images = image_batch.shape[0]
    random_number = npr.choice([0,1],num_images,replace=True)
    #the index of image needed to flip
    indexes = np.where(random_number>0)[0]
    fliplandmarkindexes = np.where(label_batch[indexes]==-2)[0]
    
    #random flip    
    for i in indexes:
        cv2.flip(image_batch[i],1,image_batch[i])
    #pay attention: flip landmark    
    for i in fliplandmarkindexes:
        landmark_ = landmark_batch[i].reshape((-1,2))
        landmark_ = np.asarray([(1-x, y) for (x, y) in landmark_])
        landmark_[[0, 1]] = landmark_[[1, 0]]#left eye<->right eye
        landmark_[[3, 4]] = landmark_[[4, 3]]#left mouth<->right mouth        
        landmark_batch[i] = landmark_.ravel()
    return image_batch,landmark_batch
'''
# all mini-batch mirror
def random_flip_images(image_batch,label_batch,landmark_batch):
    #mirror
    if random.choice([0,1]) > 0:
        num_images = image_batch.shape[0]
        fliplandmarkindexes = np.where(label_batch==-2)[0]
        flipposindexes = np.where(label_batch==1)[0]
        #only flip
        flipindexes = np.concatenate((fliplandmarkindexes,flipposindexes))
        #random flip    
        for i in flipindexes:
            cv2.flip(image_batch[i],1,image_batch[i])        
        
        #pay attention: flip landmark    
        for i in fliplandmarkindexes:
            landmark_ = landmark_batch[i].reshape((-1,2))
            landmark_ = np.asarray([(1-x, y) for (x, y) in landmark_])
            landmark_[[0, 1]] = landmark_[[1, 0]]#left eye<->right eye
            #landmark_[[3, 4]] = landmark_[[4, 3]]#left mouth<->right mouth        
            landmark_batch[i] = landmark_.ravel()
        
    return image_batch,landmark_batch

def train(net_factory, prefix, end_epoch, base_dir,
          display=200, base_lr=0.01,COLOR_GRAY = 0):   ###
    """
    train PNet/RNet/ONet
    :param net_factory:
    :param prefix:
    :param end_epoch:16
    :param dataset:
    :param display:
    :param base_lr:
    :return:
    """
    net = prefix.split('/')[-1]
    #label file
    #label_file = os.path.join(base_dir,'train_%s_32_landmarkcolor.txt' % net)
    #label_file = os.path.join(base_dir,'landmark_12_few.txt')
    #print (label_file )
    #f = open(label_file, 'r')
    num = 1600000#531806  #950000   # 1500000
    #num = len(f.readlines())
    #f.close()
    print("Total datasets is: ", num)
    print (prefix)

    if net == 'PNet' and not config.SINGLEF:
        if COLOR_GRAY ==0:
            #dataset_dir = os.path.join(base_dir,'train_%s_ALL.tfrecord_shuffle' % net)
            dataset_dir = os.path.join(base_dir,'train_%s_12_color.tfrecord_shuffle' % net)
        elif COLOR_GRAY ==1:
            dataset_dir = os.path.join(base_dir,'train_%s_12_gray.tfrecord_shuffle' % net)
        print (dataset_dir)
        #pdb.set_trace()
        image_batch, label_batch, bbox_batch,landmark_batch = read_single_tfrecord(dataset_dir, config.BATCH_SIZE, net,COLOR_GRAY)
    elif  net == 'PNet' and  config.SINGLEF:
        if COLOR_GRAY ==0:
            pos_dir = os.path.join(base_dir,'PNet_12_color_pos.tfrecord_shuffle')
            part_dir = os.path.join(base_dir,'PNet_12_color_part.tfrecord_shuffle')
            neg_dir = os.path.join(base_dir,'PNet_12_color_neg.tfrecord_shuffle')
        elif COLOR_GRAY ==1:
            pos_dir = os.path.join(base_dir,'PNet_12_gray_pos.tfrecord_shuffle')
            part_dir = os.path.join(base_dir,'PNet_12_gray_part.tfrecord_shuffle')
            neg_dir = os.path.join(base_dir,'PNet_12_gray_neg.tfrecord_shuffle')
        landmark_dir = None
        dataset_dirs = [pos_dir,part_dir,neg_dir,landmark_dir]

        pos_radio = 1.0/5;part_radio = 1.0/5;landmark_radio=1.0/6;neg_radio=3.0/5
        pos_batch_size = int(np.ceil(config.BATCH_SIZE*pos_radio))
        assert pos_batch_size != 0,"Batch Size Error "
        part_batch_size = int(np.ceil(config.BATCH_SIZE*part_radio))
        assert part_batch_size != 0,"Batch Size Error "        
        neg_batch_size = int(np.ceil(config.BATCH_SIZE*neg_radio))
        assert neg_batch_size != 0,"Batch Size Error "
        landmark_batch_size = int(np.ceil(config.BATCH_SIZE*landmark_radio))
        assert landmark_batch_size != 0,"Batch Size Error "
        batch_sizes = [pos_batch_size,part_batch_size,neg_batch_size,landmark_batch_size]
        landmarkflag = 0
        partflag = 1
        image_batch, label_batch, bbox_batch,landmark_batch = read_multi_tfrecords(dataset_dirs,batch_sizes, net,landmarkflag,partflag,COLOR_GRAY)    
    elif net == 'RNet':
        if COLOR_GRAY ==0:
            pos_dir = os.path.join(base_dir,'RNet_24_color_pos.tfrecord_shuffle')
            part_dir = os.path.join(base_dir,'RNet_24_color_part.tfrecord_shuffle')
            neg_dir = os.path.join(base_dir,'RNet_24_color_neg.tfrecord_shuffle')
            #landmark_dir = os.path.join(base_dir,'RNet_24_color_landmark.tfrecord_shuffle')
        elif COLOR_GRAY ==1:
            pos_dir = os.path.join(base_dir,'RNet_24_gray_pos.tfrecord_shuffle')
            part_dir = os.path.join(base_dir,'RNet_24_gray_part.tfrecord_shuffle')
            neg_dir = os.path.join(base_dir,'RNet_24_gray_neg.tfrecord_shuffle')
            #landmark_dir = os.path.join(base_dir,'RNet_24_gray_landmark.tfrecord_shuffle')
        landmark_dir = None
        dataset_dirs = [pos_dir,part_dir,neg_dir,landmark_dir]
        pos_radio = 1.0/5;part_radio = 1.0/5;landmark_radio=1.0/6;neg_radio=3.0/5
        pos_batch_size = int(np.ceil(config.BATCH_SIZE*pos_radio))
        assert pos_batch_size != 0,"Batch Size Error "
        part_batch_size = int(np.ceil(config.BATCH_SIZE*part_radio))
        assert part_batch_size != 0,"Batch Size Error "        
        neg_batch_size = int(np.ceil(config.BATCH_SIZE*neg_radio))
        assert neg_batch_size != 0,"Batch Size Error "
        landmark_batch_size = int(np.ceil(config.BATCH_SIZE*landmark_radio))
        assert landmark_batch_size != 0,"Batch Size Error "
        batch_sizes = [pos_batch_size,part_batch_size,neg_batch_size,landmark_batch_size]
        landmarkflag = 0   #  select landmarkflag
        partflag = 1
        image_batch, label_batch, bbox_batch,landmark_batch = read_multi_tfrecords(dataset_dirs,batch_sizes, net,landmarkflag,partflag,COLOR_GRAY)        
    elif net =='ONet':
        if COLOR_GRAY ==0:
            pos_dir = os.path.join(base_dir,'ONet_48_color_pos.tfrecord_shuffle')
            part_dir = os.path.join(base_dir,'ONet_48_color_part.tfrecord_shuffle')
            neg_dir = os.path.join(base_dir,'ONet_48_color_neg.tfrecord_shuffle')
            #landmark_dir = os.path.join(base_dir,'ONet_48_color_landmark.tfrecord_shuffle')
        elif COLOR_GRAY ==1:
            pos_dir = os.path.join(base_dir,'ONet_48_gray_pos.tfrecord_shuffle')
            #part_dir = os.path.join(base_dir,'ONet_48_gray_part.tfrecord_shuffle')
            neg_dir = os.path.join(base_dir,'ONet_48_gray_neg_single.tfrecord_shuffle')
            #landmark_dir = os.path.join(base_dir,'ONet_48_gray_landmark.tfrecord_shuffle')
        part_dir = None
        landmark_dir = None
        #part_dir = None
        dataset_dirs = [pos_dir,part_dir,neg_dir,landmark_dir]
        pos_radio = 1.0/2;part_radio = 1.0/6;landmark_radio=1.0/6;neg_radio=1.0/2
        pos_batch_size = int(np.ceil(config.BATCH_SIZE*pos_radio))
        assert pos_batch_size != 0,"Batch Size Error "
        part_batch_size = int(np.ceil(config.BATCH_SIZE*part_radio))
        assert part_batch_size != 0,"Batch Size Error "        
        neg_batch_size = int(np.ceil(config.BATCH_SIZE*neg_radio))
        assert neg_batch_size != 0,"Batch Size Error "
        landmark_batch_size = int(np.ceil(config.BATCH_SIZE*landmark_radio))
        assert landmark_batch_size != 0,"Batch Size Error "
        batch_sizes = [pos_batch_size,part_batch_size,neg_batch_size,landmark_batch_size]
        landmarkflag = 0
        partflag = 0
        image_batch, label_batch, bbox_batch,landmark_batch = read_multi_tfrecords(dataset_dirs,batch_sizes, net,landmarkflag,partflag,COLOR_GRAY)        


    #landmark_dir    
    if net == 'PNet':
        image_size = 12
        radio_cls_loss = 1.0;radio_bbox_loss = 0.5;radio_landmark_loss = 0.5
    elif net == 'RNet':
        image_size = 24
        radio_cls_loss = 1.0;radio_bbox_loss = 0.5;radio_landmark_loss = 0.5
    else:
        radio_cls_loss = 1.0;radio_bbox_loss = 0;radio_landmark_loss = 0
        image_size = 48
    
    #define placeholder
    if COLOR_GRAY == 1:
        input_image = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, image_size, image_size, 1], name='input_image')
    else:
        input_image = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, image_size, image_size, 3], name='input_image')
    label = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE], name='label')
    bbox_target = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, 4], name='bbox_target')
    landmark_target = tf.placeholder(tf.float32,shape=[config.BATCH_SIZE,4],name='landmark_target')
    #class,regression
    print('class,regression+')
    '''
    cls_loss_op,bbox_loss_op,landmark_loss_op,L2_loss_op,accuracy_op = net_factory(
        input_image, label, bbox_target,landmark_target,training=True)
    '''
    cls_loss_op,L2_loss_op,accuracy_op = net_factory(
        input_image, label, bbox_target,landmark_target,training=True)
    #train,update learning rate(3 loss)
    # train_op, lr_op,global_step = train_model(base_lr, radio_cls_loss*cls_loss_op + radio_bbox_loss*bbox_loss_op + radio_landmark_loss*landmark_loss_op + L2_loss_op, num)
    train_op, lr_op,global_step = train_model(base_lr, radio_cls_loss*cls_loss_op +  L2_loss_op, num)
    # init
    init = tf.global_variables_initializer()
    ###gpu
    #configp = tf.ConfigProto()
    #configp.allow_soft_placement = True
    #configp.gpu_options.per_process_gpu_memory_fraction = 0.3
    #configp.gpu_options.allow_growth = True
    #sess = tf.Session(config =configp)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) 
    #save model
    saver = tf.train.Saver(tf.trainable_variables(),max_to_keep=3)
    #saver = tf.train.Saver(max_to_keep=3)
    # restore model
    if  net == 'PNet':
        if COLOR_GRAY == 1:
            pretrained_model = './data/new_model/PNet_merge_gray'
        else:
            pretrained_model = './data/new_model/PNet_merge_color'
    elif net =='RNet':
        if COLOR_GRAY == 1:
            pretrained_model = './data/new_model/RNet_cmerge_gray_gray'
        else:
            pretrained_model = './data/new_model/RNet_cmerge_gray_color'
    elif net =='ONet':
        pretrained_model = './data/new_model/test1-1ONet_NIR_calib_A_gray'
    else:
        pretrained_model =None


    sess.run(init)
    print(sess.run(global_step))
    if pretrained_model and config.PRETRAIN:
        print('Restoring pretrained model: %s' % pretrained_model)
        ckpt = tf.train.get_checkpoint_state(pretrained_model)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Not Pretrain \n')
    
    print(sess.run(global_step))
    #visualize some variables
    tf.summary.scalar("cls_loss",cls_loss_op)#cls_loss
    #tf.summary.scalar("bbox_loss",bbox_loss_op)#bbox_loss
   # tf.summary.scalar("landmark_loss",landmark_loss_op)#landmark_loss
    tf.summary.scalar("cls_accuracy",accuracy_op)#cls_acc
    summary_op = tf.summary.merge_all()
    logs_dir = "./logs/%s" %(net)
    if os.path.exists(logs_dir) == False:
        os.mkdir(logs_dir)
    writer = tf.summary.FileWriter(logs_dir,sess.graph)
    #begin 
    coord = tf.train.Coordinator()
    #begin enqueue thread
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    i = 0
    #total steps
    MAX_STEP = int(num / config.BATCH_SIZE + 1) * end_epoch
    
    epoch = 0
    sess.graph.finalize()    
    try:
        for step in range(MAX_STEP):
            i = i + 1
            if coord.should_stop():
                break
            image_batch_array, label_batch_array, bbox_batch_array,landmark_batch_array = sess.run([image_batch, label_batch, bbox_batch,landmark_batch])
            #random flip
            #image_batch_array,landmark_batch_array = random_flip_images(image_batch_array,label_batch_array,landmark_batch_array)
            '''
            print image_batch_array.shape
            print label_batch_array.shape
            print bbox_batch_array.shape
            print landmark_batch_array.shape
            print label_batch_array[0]
            print bbox_batch_array[0]
            print landmark_batch_array[0]
            '''
            _,_,summary = sess.run([train_op, lr_op ,summary_op], feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array,landmark_target:landmark_batch_array})
            
            if (step+1) % display == 0:
                
                #acc = accuracy(cls_pred, labels_batch)
                cls_loss, L2_loss,lr,acc = sess.run([cls_loss_op,L2_loss_op,lr_op,accuracy_op],
                                                             feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array, landmark_target: landmark_batch_array})                
                print("%s : Step: %d, accuracy: %3f, cls loss: %4f, L2 loss: %4f,lr:%f " % (
                datetime.now(), step+1, acc, cls_loss, L2_loss, lr))
                
                
            #save every two epochs
            if i * config.BATCH_SIZE > num:
                epoch = epoch + 1
                print('save epoch%d'%epoch)
                i = 0
                saver.save(sess, prefix, global_step=epoch)
            writer.add_summary(summary,global_step=step)
    except tf.errors.OutOfRangeError:
        print("完成！！！")
    finally:
        coord.request_stop()
        writer.close()
    coord.join(threads)
    sess.close()
