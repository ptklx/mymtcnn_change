import sys
#sys.path.append('../')
sys.path.append('X:/deeplearn/mtcnn/MTCNN-Tensorflow-masterP')
sys.path.append('X:/deeplearn/mtcnn/MTCNN-Tensorflow-masterP/Detection')
sys.path.append('X:/deeplearn/mtcnn/MTCNN-Tensorflow-masterP/prepare_data')
from collections import Iterable

from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from train_models.mtcnn_model import P_Net, P_Net_new,R_Net,O_Net,O_Net_new
from prepare_data.loader import TestLoader
import cv2
import os
import numpy as np
test_mode = "ONet"
thresh = [0.6, 0.5, 0.3]
min_face_size = 40    #  24
stride = 2
slide_window = False
shuffle = False
detectors = [None, None, None]
#prefix = ['../data/MTCNN_model/PNet_landmark/PNet', '../data/MTCNN_model/RNet_landmark/RNet', '../data/MTCNN_model/ONet_landmark/ONet']
#prefix = ['X:/deeplearn/mtcnn/MTCNN-Tensorflow-master_change/data/MTCNN_model/PNet_landmarktest/PNet']

prefix = ['X:/deeplearn/mtcnn/MTCNN-Tensorflow-master_change/data/new_model/PNet_NIR_calib_gray/PNet', 
          'X:/deeplearn/mtcnn/MTCNN-Tensorflow-master_change/data/new_model/RNet_NIR_calib_gray/RNet',
          'X:/deeplearn/mtcnn/MTCNN-Tensorflow-master_change/data/new_model/test2ONet_NIR_calib_gray/ONet']
epoch = [30,40,40]  # train select
COLOR_GRAY = 1  #color 0 gray 1


batch_size = [2048,256,16]
model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
# load pnet model
if slide_window:
    PNet = Detector(P_Net, 12, batch_size[0], model_path[0],COLOR_GRAY)
else:
    #PNet = FcnDetector(P_Net_new, model_path[0])
    PNet = FcnDetector(P_Net, model_path[0],COLOR_GRAY)
detectors[0] = PNet
# load rnet model
if test_mode in ["RNet", "ONet"]:
    RNet = Detector(R_Net, 24, batch_size[1], model_path[1],COLOR_GRAY)
    detectors[1] = RNet

# load onet model
if test_mode == "ONet":
    ONet = Detector(O_Net_new, 48, batch_size[2], model_path[2],COLOR_GRAY)
    detectors[2] = ONet


mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh, slide_window=slide_window,COLO_GRA = COLOR_GRAY)
gt_imdb = []
#gt_imdb.append("35_Basketball_Basketball_35_515.jpg")
#imdb_ = dict()"
#imdb_['image'] = im_path
#imdb_['label'] = 5
path = "X:\\deeplearn\\mtcnn\\MTCNN-Tensorflow-master_change\\test\\lala"
for item in os.listdir(path):
    if item.find('.png') == -1 and item.find('.bmp') ==-1 and item.find('.jpg')==-1:
        continue
    gt_imdb.append(os.path.join(path,item))
test_data = TestLoader(gt_imdb, batch_size=1, shuffle=False,COLO_GRA = COLOR_GRAY)
#all_boxes = mtcnn_detector.detect_facePnet(test_data)
all_boxes ,landmarks= mtcnn_detector.detect_face(test_data)
count = 0
for imagepath in gt_imdb:
    print (imagepath)
    image = cv2.imread(imagepath)
    h = image.shape[0]
    w = image.shape[1]
    if h == 640 and w ==480:
        image = cv2.resize(image,(480,572))
    for bbox in all_boxes[count]:
        cv2.putText(image,str(np.round(bbox[4],2)),(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255))
        cv2.rectangle(image, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255))
    if test_mode in [ "ONet"]:#isinstance(landmarks,Iterable):
        for landmark in landmarks[count]:
            if isinstance(landmark,Iterable):
                for i in range(int(len(landmark)/2)):
                    cv2.circle(image, (int(landmark[2*i]),int(landmark[2*i+1])), 3, (0,0,255))


    count = count + 1
    #cv2.imwrite("result_landmark/%d.png" %(count),image)
    filename = os.path.basename(imagepath)
    cv2.imshow("lala",image)
    cv2.waitKey(0)    

'''
for data in test_data:
    print type(data)
    for bbox in all_boxes[0]:
        print bbox
        print (int(bbox[0]),int(bbox[1]))
        cv2.rectangle(data, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255))
    #print data
    cv2.imshow("lala",data)
    cv2.waitKey(0)
'''