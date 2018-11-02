#coding:utf-8
from mtcnn_model import O_Net,O_Net_new,O_Net_new1
from train import train
from MTCNN_config import config

def train_ONet(base_dir, prefix, end_epoch, display, lr,COLOR_GRA):
    """
    train PNet
    :param dataset_dir: tfrecord path
    :param prefix:
    :param end_epoch:
    :param display:
    :param lr:
    :return:
    """
    net_factory = O_Net_new1 #O_Net
    train(net_factory, prefix, end_epoch, base_dir, display=display, base_lr=lr,COLOR_GRAY=COLOR_GRA)

if __name__ == '__main__':
    '''
    base_dir ='X:\\deeplearn\\mtcnn\\MTCNN-Tensorflow-masteP\\imglists\\ONet_24'
    model_path = 'X:/deeplearn/mtcnn/MTCNN-Tensorflow-masterP/data/MTCNN_model/ONet_landmarkcolor/ONet'
    '''
    COLOR_GRAY = 1#  color 0 gray 1
    Mark = 'NIR_calib_A'
    #base_dir = '../MTCNN-Tensorflow-master_change/imglists/OnetA/%s'%Mark
    base_dir = './imglists/OnetA/%s'%Mark
    if COLOR_GRAY == 0:
        model_name = 'color'
    else:
        model_name = 'gray'
    if config.PRETRAIN == 1:
        model_path = './data/pre_model/single1-1ONet_%s_%s/ONet'%(Mark,model_name)   #savemodle path
    else:
        model_path = './data/new_model/single1-2ONet_%s_%s/ONet'%(Mark,model_name)   #savemodle path

    prefix = model_path
    end_epoch = 40
    display = 500
    if config.PRETRAIN == 1:
        lr = 0.005
    else:
        lr = 0.01
    train_ONet(base_dir, prefix, end_epoch, display, lr,COLOR_GRAY)