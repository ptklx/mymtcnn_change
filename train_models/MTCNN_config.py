#coding:utf-8

from easydict import EasyDict as edict

config = edict()

config.BATCH_SIZE = 360
config.CLS_OHEM = True
config.CLS_OHEM_RATIO = 0.7
config.BBOX_OHEM = False
config.BBOX_OHEM_RATIO = 0.7

config.EPS = 1e-14
#config.LR_EPOCH = [6,14,20,30,50]
config.LR_EPOCH = [3,7,14,20,30,50]
#config.COLOR_GRAY = 0   # color 0 gray 1
config.PRETRAIN = 1   #restore pretrain
config.SINGLEF =1 #