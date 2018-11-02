#coding:utf-8
import sys
import os
#print (os.path.dirname(os.path.abspath(__file__)))
sys.path.append('O:/deeplearn/mtcnn/MTCNN-Tensorflow-masterP')
sys.path.append('O:/deeplearn/mtcnn/MTCNN-Tensorflow-masterP/Detection')
sys.path.append('O:/deeplearn/mtcnn/MTCNN-Tensorflow-masterP/prepare_data')
sys.path.append('/home/pengtao/deeplearn/mtcnn/MTCNN-Tensorflow-masterP')
#from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from train_models.mtcnn_model import P_Net,  O_Net,O_Net_new,O_Net_new1,O_Net_new2
from nms import py_nms
import cv2
import numpy as np
#from prepare_data.utils import IoU
test_mode = "ONet"
thresh = [0.6, 0.5, 0.9]
min_face_size = 60
stride = 2
slide_window = False
shuffle = False
#vis = True
detectors = [None, None, None]


COLOR_GRAY = 1  #color 0 gray 1
#name = 'RNet_color_real'  ########
name = 'NIR_calib_ONet1' 
#win
prefix = ['O:/deeplearn/mtcnn/MTCNN-Tensorflow-master_change/data/new_model/PNet_NIR_calib_gray/PNet', 
          'O:/deeplearn/mtcnn/MTCNN-Tensorflow-masterP/data/new_model/RNet_NIR_calib_gray/RNet',
          'O:/deeplearn/mtcnn/MTCNN-Tensorflow-masterP/data/pre_model/single1-1ONet_NIR_calib_A_gray/ONet']
#O_Net_new  test3 ONet_NIR_calib_gray/ONet

epoch = [30, 40,40]

net_factory = O_Net_new1

picpath = 'V:/NIR_ALL_calib/NIR_labelpic'
txtpath = 'V:/NIR_ALL_calib/NIR_labelpic_landmark.txt'


txtnpdpath = 'V:/NIR_ALL_calib/NIR_labelpic_NPD_detection_coor_new.txt'

Mpath =  'R:\\pengtao\\cropImage\\Onet1'
negpath = 'R:\\pengtao\\cropImage\\Onet1\\neg\\%s'%name
pospath = 'R:\\pengtao\\cropImage\\Onet1\\pos\\%s'%name

ourcamera = 1

#imagepath = 'O:\\deeplearn\\mtcnn\\MTCNN-Tensorflow-masterP\\test\\trainimg'
#imagepath = "O:\\deeplearn\\mtcnn\\MTCNN-Tensorflow-masterP\\test\\wrong"
#imagepath = "O:\\deeplearn\\mtcnn\\MTCNN-Tensorflow-masterP\\test\\right"

imagepath = None# 'D:\\win7_favorite\\Desktop\\faceReal\\1'

#linux

'''
txtpath = '/datahome/facenet/NIR_ALL_calib/NIR_labelpic_landmark.txt'
picpath = '/datahome/facenet/NIR_ALL_calib/NIR_labelpic'
ourcamera  = 1
#txtpath = '/home/pengtao/deeplearn/mtcnn/MTCNN_data/color_5dotin.txt'

Mpath = '/home/pengtao/deeplearn/mtcnn/cropImage/pretrain/%s'%name
negpath = '/home/pengtao/deeplearn/mtcnn/cropImage/pretrain/%s/neg'%name
pospath = '/home/pengtao/deeplearn/mtcnn/cropImage/pretrain/%s/pos'%name
'''

if not os.path.exists(Mpath):
    os.makedirs(Mpath)
if not os.path.exists(negpath):
    os.makedirs(negpath)
if not os.path.exists(pospath):
        os.makedirs(pospath)


model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]

PNet = FcnDetector(P_Net, model_path[0],COLOR_GRAY)
detectors[0] = PNet

ONet = Detector(net_factory, 48, 1, model_path[2],COLOR_GRAY,MODE = test_mode)
detectors[2] = ONet

#mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               #stride=stride, threshold=thresh, slide_window=slide_window,COLO_GRA = COLOR_GRAY)

def main():
    
    if imagepath != None: 
        picpathlist  = getpiclist1(imagepath)
    elif txtnpdpath != None:
        picpathlist ,faceboxlist = getpiclist3(txtnpdpath)
    else:
        picpathlist ,faceboxlist = getpiclist2(txtpath,ourcamera)

    npdeyenum = 0
    mtcnnnum = 0
    npdfalsenum = 0
    falsenum = 0
    ftxt = open('negtxt.txt','w')
    picNum = len(picpathlist)
    for iNum in range(picNum):
        if imagepath != None: 
            image_path=picpathlist[iNum]
        else:
            image_path = os.path.join(picpath,picpathlist[iNum])
        if imagepath == None:
            facebox = faceboxlist[iNum] 
        
        if COLOR_GRAY==0:
            image = cv2.imread(image_path,cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        if COLOR_GRAY==0:
            cropped_ims = np.zeros((1, 48, 48, 3), dtype=np.float32)
            if imagepath != None:
                cropimg = image
            else:
                cropimg = image[facebox[1] : facebox[3], facebox[0] : facebox[2], :]           
        elif COLOR_GRAY==1:
            cropped_ims = np.zeros((1, 48, 48, 1), dtype=np.float32)
            if imagepath != None:
                cropimg = image
            else:  
                cropimg = image[facebox[1] : facebox[3], facebox[0] : facebox[2]]
        #= cv2.resize(cropimg, (48, 48), interpolation=cv2.INTER_LINEAR)
        if txtnpdpath != None:
            if facebox[0]==0 or cropimg is None or facebox[4]==0:
                npdfalsenum+=1
                continue

        cropped_ims[0, :, :, 0] = (cv2.resize(cropimg, (48, 48))-127.5) / float(128)
        #safef = open('input_data_v2.bin', "wb")
        #safef.write(cropped_ims)
        #safef.close()
        if imagepath != None:
            cls_scores = detectors[2].predict(cropped_ims)
            cls_scores = cls_scores[0][0][1]  
            if cls_scores<thresh[2]:
                #print('not detect face!')
                print(cls_scores)
                cv2.imshow("neg",image)
                #cv2.imshow('croop',cropimg)
                cv2.waitKey()
                continue
            print('right score:%f'%cls_scores)
            print(image_path)
            cv2.imshow("personOneFace",image)
            cv2.waitKey()
        else:
            cls_scores = detectors[2].predict(cropped_ims)
            #if facebox[4]==0:

            #prob belongs to face
            cls_scores = cls_scores[0][0][1]                
            if cls_scores < thresh[2]:
                #print('not detect face!')
                #cv2.imshow("neg",image)
                #cv2.imshow('croop',cropimg)
                #cv2.waitKey()
                print('false score:%f'%cls_scores)
                ftxt.write(str(cls_scores))
                ftxt.write('\n')
                falsenum+=1
                save_file = os.path.join(pospath, "%s" % picpathlist[iNum])
                cv2.imwrite(save_file, cropimg)
                continue
               
            #if facebox[4] == 0:
                #npdeyenum+=1
            mtcnnnum+=1
            #print('right score:%f'%cls_scores)
            '''
            #width
            w = facebox[2]-facebox[0]
            #height 
            h = facebox[3]-facebox[1]
            temp = landmark
            landmark[0,0::2] = (np.tile(w,(2,1)) * temp[0,0::2].T  - 1+facebox[0])[0].T

            landmark[0,1::2] = (np.tile(h,(2,1)) * temp[0,1::2].T  - 1+facebox[1])[0].T
            landmark = temp[0]
            for i in range(int(len(landmark)/2)):
                cv2.circle(image, (int(landmark[2*i]),int(landmark[2*i+1])), 3, (255,255,255))
            #cv2.imshow("personOneF",image)
            cv2.putText(image,str(np.round(cls_scores[0],2)),(int(facebox[0]),int(facebox[1])),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255))
            cv2.rectangle(image,(facebox[0],facebox[1]),(facebox[2],facebox[3]),(255,0,255),1)
            '''
            #cv2.imshow("personOneFace",image)
            #cv2.waitKey()

        #showimg(image, facebox,0)
        #cv2.waitKey()
        #boxes_c,_ = mtcnn_detector.detect(image)
        #save_file = os.path.join(pospath, "%s" % picpathlist[iNum])
        #cv2.imwrite(save_file, img)
        #savepospicN+=1
    print('all pic num: %d'%picNum)
    print('npdfalsenum false num %d',npdfalsenum)
    print('mtcnn detct face num %d'%mtcnnnum)
    print('npd not detect eye num: %d'%npdeyenum)
    print('false %d'%falsenum)
    ftxt.close()
def main2():
    #picpathlist ,faceboxlist = getpiclist2(txtpath,ourcamera)
    #pathtest = "X:\\deeplearn\\mtcnn\\MTCNN-Tensorflow-master_change\\test\\wrong"
    picpathlist  = getpiclist1(None)
    picNum = len(picpathlist)
    for iNum in range(picNum):
        image_path = os.path.join(picpath,picpathlist[iNum])
        # = faceboxlist[iNum]
        
        if COLOR_GRAY==0:
            image = cv2.imread(image_path,cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        image_color = cv2.imread(image_path,cv2.IMREAD_COLOR)
        h, w = image.shape
        if h == 640 and w ==480:
            image_color = cv2.resize(image_color,(480,572))
            image = cv2.resize(image,(480,572))
        if image is None:
            continue
        _, dets, _ = detect_pnet(detectors[0], image,60,thresh[0])
       
        if len(image.shape) == 3:
            h, w, c = image.shape
        else:
            h, w = image.shape
        if dets is None:
            print('Pnet detect is None')
            cv2.imshow("neg_img",image_color)
            cv2.waitKey()
            continue
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
        num_boxes = dets.shape[0]
        if COLOR_GRAY == 0:
            cropped_ims = np.zeros((num_boxes, 48, 48, 3), dtype=np.float32)
            for i in range(num_boxes):
                tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
                tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = image[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
                cropped_ims[i, :, :, :] = (cv2.resize(tmp, (48, 48))-127.5) / float(128)
        elif COLOR_GRAY == 1:
            cropped_ims = np.zeros((num_boxes, 48, 48, 1), dtype=np.float32)
            for i in range(num_boxes):
                tmp = np.zeros((tmph[i], tmpw[i], 1), dtype=np.uint8)
                tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1,0] = image[y[i]:ey[i] + 1, x[i]:ex[i] + 1]
                cropped_ims[i, :, :, 0] = (cv2.resize(tmp, (48, 48))-127.5) / float(128)

        cls_scores = detectors[2].predict(cropped_ims)
        #prob belongs to face
        len_score = len(cls_scores)  

        cls_scores = cls_scores.reshape([len_score,2])

        cls_scores= cls_scores[:,1]
        keep_inds = np.where(cls_scores > thresh[2])[0]   
        if len(keep_inds) > 0:
            #pickout filtered box
            all_boxes = dets[keep_inds]
            all_boxes[:, 4] = cls_scores[keep_inds]
          
        else:
            print('not detect face!')
            for bbox in dets:
                cv2.putText(image_color,str(np.round(bbox[4],2)),(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255))
                cv2.rectangle(image_color, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255))
            cv2.imshow("neg_img",image_color)
            #cv2.imshow('croop',cropimg)
            cv2.waitKey()
            continue

        '''
        temp = landmark
        landmark[0,0::2] = (np.tile(w,(2,1)) * temp[0,0::2].T  - 1+facebox[0])[0].T

        landmark[0,1::2] = (np.tile(h,(2,1)) * temp[0,1::2].T  - 1+facebox[1])[0].T
        landmark = temp[0]
        for i in range(int(len(landmark)/2)):
            cv2.circle(image, (int(landmark[2*i]),int(landmark[2*i+1])), 3, (255,255,255))
        #cv2.imshow("personOneF",image)
        cv2.putText(image,str(np.round(cls_scores[0],2)),(int(facebox[0]),int(facebox[1])),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255))
        cv2.rectangle(image,(facebox[0],facebox[1]),(facebox[2],facebox[3]),(255,0,255),1)
        '''
        for bbox in all_boxes:
            cv2.putText(image_color,str(np.round(bbox[4],2)),(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255))
            cv2.rectangle(image_color, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255))
        compareImage = image_color.copy()
        for bbox in dets:
            cv2.putText(compareImage,str(np.round(bbox[4],2)),(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255))
            cv2.rectangle(compareImage, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255))
        cv2.imshow('compareImage_Pnet',compareImage)
        cv2.imshow("personOneFace_Onet",image_color)
        cv2.waitKey()

def pad( bboxes, w, h):
        """
            pad the the bboxes, alse restrict the size of it
        Parameters:
        ----------
            bboxes: numpy array, n x 5
                input bboxes
            w: float number
                width of the input image
            h: float number
                height of the input image
        Returns :
        ------
            dy, dx : numpy array, n x 1
                start point of the bbox in target image
            edy, edx : numpy array, n x 1
                end point of the bbox in target image
            y, x : numpy array, n x 1
                start point of the bbox in original image
            ex, ex : numpy array, n x 1
                end point of the bbox in original image
            tmph, tmpw: numpy array, n x 1
                height and width of the bbox
        """
        tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + 1, bboxes[:, 3] - bboxes[:, 1] + 1
        num_box = bboxes.shape[0]

        dx, dy = np.zeros((num_box,)), np.zeros((num_box,))
        edx, edy = tmpw.copy() - 1, tmph.copy() - 1

        x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

        tmp_index = np.where(ex > w - 1)
        edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
        ex[tmp_index] = w - 1

        tmp_index = np.where(ey > h - 1)
        edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
        ey[tmp_index] = h - 1

        tmp_index = np.where(x < 0)
        dx[tmp_index] = 0 - x[tmp_index]
        x[tmp_index] = 0

        tmp_index = np.where(y < 0)
        dy[tmp_index] = 0 - y[tmp_index]
        y[tmp_index] = 0

        return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
        return_list = [item.astype(np.int32) for item in return_list]

        return return_list

def convert_to_square(bbox):
    """
        convert bbox to square
    Parameters:
    ----------
        bbox: numpy array , shape n x 5
            input bbox
    Returns:
    -------
        square bbox
    """
    square_bbox = bbox.copy()

    h = bbox[:, 3] - bbox[:, 1] + 1
    w = bbox[:, 2] - bbox[:, 0] + 1
    max_side = np.maximum(h, w)
    square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
    square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
    square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
    return square_bbox

def getpiclist(pictxt):
    idx = 0
    path_list = []
    picmark_list= []
    with open(pictxt, 'r') as f:
        while True:
            annotation = f.readline()
            if not annotation:
                break
            if idx ==0:
                idx += 1
                continue
            idx += 1
            annotation = annotation.strip().split()
            path_list.append(annotation[0])
            del annotation[0]
            onemark =[]
            for index , x in enumerate(annotation):
                if index%2 ==1:
                    b = int(float(572.0/640.0) *float(x))
                else:
                    b = int(float(x))
                onemark.append(b)
            picmark_list.append(onemark)
    return path_list,picmark_list

def getpiclist1(picpath):
    idx = 0
    path_list = []
    if picpath == None:
        picpath = "O:\\deeplearn\\mtcnn\\MTCNN-Tensorflow-masterP\\test\\lala"
    for item in os.listdir(picpath):
        if item.find('.png') == -1 and item.find('.bmp') ==-1 and item.find('.jpg')==-1:
            continue
        path_list.append(os.path.join(picpath,item))
    return path_list


def getpiclist2(pictxt,outcamera = 0):   #原始坐标，minx, miny ,maxx,maxy
    path_list = []
    picmark_list= []
    with open(pictxt, 'r') as f:
        while True:
            annotation = f.readline()
            if not annotation:
                break
            annotation = annotation.strip().split()
            path_list.append(annotation[0])
            xratio = 1#0.6
            yratio = 0.7#0.35
            if outcamera:
                landmarks = np.array(annotation[1:11], dtype=np.float32).reshape(-1, 10)
                #print(landmarks)
                x0 = landmarks[:,0]-xratio*(landmarks[:,2] - landmarks[:,0])/2.0
                x0 = x0 if x0>0 else np.array([0],dtype=np.float32) 
                #x0 = np.where(x0<0,0,x0)
                y0 = (landmarks[:,1] + landmarks[:,3]+yratio*(landmarks[:,1] + landmarks[:,3] - landmarks[:,7]-landmarks[:,9]))/2.0
                y0 = y0 if y0>0 else np.array([0],dtype=np.float32) 
                #y0 = np.where(y0<0,0,y0)
                x1 = landmarks[:,2]+xratio*(landmarks[:,2] - landmarks[:,0])/2.0  
                x1 = x1 if x1< 479 else np.array([479.0],dtype=np.float32)      
                y1 =(landmarks[:,7] + landmarks[:,9]-yratio*(landmarks[:,1] + landmarks[:,3] - landmarks[:,7]-landmarks[:,9]))/2.0            
                y1 = y1 if y1 < 571 else np.array([571.0],dtype=np.float32) 
                temp = [x0,y0,x1,y1]
                if (y1-y0)>(x1-x0):
                    size = y1 - y0
                    x0 = x0 -(size - (x1 - x0))/2.0
                    x1 = x0+size
                    if x0 < 0 or x1 > 479:
                        x0 = temp[0]
                        x1 = temp[2]
                        size = x1-x0
                        y0 = y0+(y1-y0-size)/2.0
                        y1 = y0+size
                else:
                    size = x1 - x0
                    y0 = y0 - (size-(y1-y0))/2.0
                    y1 = y0+size
                    if y0<0 or y1>571:
                        y0 = temp[1]
                        y1 = temp[3]
                        size = y1-y0
                        x0= x0+(x1-x0-size)/2.0
                        x1 = x0+size
                #x1= np.where(x1>479,479,x1)
                #y1= np.where(y1>572,572,y1)
                if x0<0 or y0<0 or x1>479 or y1>571:
                    print(x0,y0,x1,y1,size)
                    print(landmarks)
                    print(annotation[0])
                    print('error')
                    #continue
                #boxes= np.concatenate((x0,y0,x1,y1),axis = 0).reshape(-1, 4)
               
                onemark =list(map(int,[x0,y0,x1,y1]))
            else:
                onemark =list(map(int,annotation[1:5]))
            
            picmark_list.append(onemark)
    return path_list,picmark_list


#filepath face.col_min face.row_min face.col_max face.row_max eyes_L.col eyes_L.row eyes_R.col eyes_R.row
def getpiclist3(pictxt):   # 
    idx = 0
    path_list = []
    picmark_list= []
    with open(pictxt, 'r') as f:
        while True:
            annotation = f.readline()
            if not annotation:
                break
            if idx ==0:
                idx += 1
                continue
            idx += 1
            annotation = annotation.strip().split()
            path_list.append(annotation[0])
            del annotation[0]
            onemark =[]
            for index , x in enumerate(annotation):
                b = int(float(x))
                onemark.append(b)
            picmark_list.append(onemark)

    return path_list,picmark_list

def IoUT(box, boxes,flag = 0):
    """Compute IoU between detect box and gt boxes

    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    if flag ==1:
        area = (boxes[ 2] ) * (boxes[3] )
        xx1 = np.maximum(box[0], boxes[ 0])
        yy1 = np.maximum(box[1], boxes[1])
        xx2 = np.minimum(box[2], boxes[ 2]+boxes[ 0]-1)
        yy2 = np.minimum(box[3], boxes[ 3]+boxes[ 1]-1)
    else:
        area = (boxes[2] - boxes[0] + 1) * (boxes[3] - boxes[ 1] + 1)
        xx1 = np.maximum(box[0], boxes[0])
        yy1 = np.maximum(box[1], boxes[1])
        xx2 = np.minimum(box[2], boxes[2])
        yy2 = np.minimum(box[3], boxes[3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h

    ovr = float(inter) / float(box_area + area - inter)
   
    return ovr

def detect_pnet(pnet_detector, im,min_face_size,thresh):
        """Get face candidates through pnet

        Parameters:
        ----------
        im: numpy array
            input image array

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_c: numpy array
            boxes after calibration
        """
        if len(im.shape) == 3:
            h, w, c = im.shape
        else:
            h, w = im.shape
        net_size = 12
        
        current_scale = float(net_size) / min_face_size  # find initial scale
        # print("current_scale", net_size, self.min_face_size, current_scale)
        im_resized = processed_image(im, current_scale)
        if len(im_resized.shape) == 3:
            current_height, current_width, _ = im_resized.shape
        else:
            current_height, current_width= im_resized.shape
        # fcn
        all_boxes = list()
        while min(current_height, current_width) > net_size:
            #return the result predicted by pnet
            #cls_cls_map : H*w*2
            #reg: H*w*4
            cls_cls_map, reg = pnet_detector.predict(im_resized)
            #boxes: num*9(x1,y1,x2,y2,score,x1_offset,y1_offset,x2_offset,y2_offset)
            boxes = generate_bbox(cls_cls_map[:, :,1], reg, current_scale, thresh)

            current_scale *= 0.79
            im_resized = processed_image(im, current_scale)
            if len(im_resized.shape)==3:
                current_height, current_width, _ = im_resized.shape
            else:
                current_height, current_width = im_resized.shape

            if boxes.size == 0:
                continue
            keep = py_nms(boxes[:, :5], 0.5, 'Union')
            boxes = boxes[keep]
            all_boxes.append(boxes)

        if len(all_boxes) == 0:
            return None, None,None

        all_boxes = np.vstack(all_boxes)

        # merge the detection from first stage
        keep = py_nms(all_boxes[:, 0:5], 0.7, 'Union')
        all_boxes = all_boxes[keep]
        boxes = all_boxes[:, :5]

        bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1

        # refine the boxes
      
        boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                             all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                             all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                             all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                             all_boxes[:, 4]])
        '''
        boxes_c = np.vstack([all_boxes[:, 0],
                        all_boxes[:, 1] ,
                        all_boxes[:, 2] ,
                        all_boxes[:, 3] ,
                        all_boxes[:, 4]])
        '''
        boxes_c = boxes_c.T

        return boxes, boxes_c,None
def processed_image( img, scale):
    if len(img.shape)==3:
        height, width, channels = img.shape
    else:
        height, width = img.shape
    new_height = int(height * scale)  # resized new height
    new_width = int(width * scale)  # resized new width
    new_dim = (new_width, new_height)
    img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)  # resized image
    img_resized = (img_resized - 127.5) / 128
    return img_resized

def generate_bbox( cls_map, reg, scale, threshold):
        """
            generate bbox from feature cls_map
        Parameters:
        ----------
            cls_map: numpy array , n x m 
                detect score for each position
            reg: numpy array , n x m x 4
                bbox
            scale: float number
                scale of this detection
            threshold: float number
                detect threshold
        Returns:
        -------
            bbox array
        """
        stride = 2
        #stride = 4
        cellsize = 12
        #cellsize = 25

        t_index = np.where(cls_map > threshold)   #face 

        # find nothing
        if t_index[0].size == 0:
            return np.array([])
        #offset
        dx1, dy1, dx2, dy2 = [reg[t_index[0], t_index[1], i] for i in range(4)]

        reg = np.array([dx1, dy1, dx2, dy2])
        score = cls_map[t_index[0], t_index[1]]
        boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),
                                 np.round((stride * t_index[0]) / scale),
                                 np.round((stride * t_index[1] + cellsize) / scale),
                                 np.round((stride * t_index[0] + cellsize) / scale),
                                 score,
                                 reg])

        return boundingbox.T

def showimg(img, bbxy,color):
    #img = cv2.resize(img, (480, 572), interpolation=cv2.INTER_LINEAR)
    if color:
        cv2.rectangle(img,(bbxy[0],bbxy[1]),(bbxy[2],bbxy[3]),(0,255,255),1)
    else:
        cv2.rectangle(img,(bbxy[0],bbxy[1]),(bbxy[2],bbxy[3]),(255,0,255),1)
    cv2.imshow("personOneFace",img)
    cv2.waitKey()


if __name__ == '__main__':
    main()
    #main2()