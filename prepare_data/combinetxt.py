#coding:utf-8
import sys
import numpy as np
import cv2
import os
import numpy.random as npr
from utils import IoU
anno_bboxfile = 'X:/data/CelebA/Anno/list_bbox_celeba.txt'
anno_landmarkfile = 'X:/data/CelebA/Anno/list_landmarks_celeba.txt'
merge_bbox_landmark = 'X:/data/CelebA/Anno/list_boxlandmark_celeba.txt'

#merge bbox and landmarks
'''
allnum = 0
fm = open(merge_bbox_landmark, 'w')
with open(anno_bboxfile, 'r') as fb :
    with open(anno_landmarkfile, 'r') as fl:
        while True:
            annobbox = fb.readline()
            if not annobbox:
                break
            annolandmark = fl.readline()
            if not annolandmark:
                break
            if allnum ==0 :
                if annobbox != annolandmark:
                    break
                else:
                    fm.write("%s"%(annobbox))
            elif allnum == 1 :
                annobbox = annobbox.strip().split()
                fm.write("%s"%(annobbox[0]))
                for istr in annobbox[1:]:
                    fm.write("  %s"%(istr))
                fm.write("  %s"%(annolandmark))
            else:
                annobbox = annobbox.strip().split()
                annolandmark = annolandmark.strip().split()
                if annolandmark[0] != annobbox[0]:
                    break
                fm.write("%s"%(annobbox[0]))
                for istr in annobbox[1:]:
                    fm.write("  %s"%(istr))
                for istr in annolandmark[1:]:
                    fm.write("  %s"%(istr))
                fm.write("\n")
            allnum+=1
            #if allnum >20:
                #break

fm.close()
'''

#pc
'''
imgsize = 48
im_dir = 'X:/deeplearn/mtcnn/cropImage/CelebA_%s'%imgsize
anno_file = 'X:/data/CelebA/Anno/list_bbox_celeba.txt'
homepic = "X:/data/CelebA/Img/img_celeba.7z/img_celeba"
ourcameraflag = 0
'''


#linux

imgsize = 48
im_dir = '/home/pengtao/deeplearn/mtcnn/cropImage/singleCelebA_%s'%imgsize
#anno_file = '/home/pengtao/data/CelebA/Anno/list_bbox_celeba.txt'
anno_file = '/home/pengtao/data/CelebA/Anno/singleface_landmark_celeba.txt'
homepic = "/home/pengtao/data/CelebA/Img/img_celeba.7z/img_celeba"
ourcameraflag = 0


#im_dir = 'X:/deeplearn/mtcnn/cropImage/infr_32'
#anno_file ='X:/deeplearn/mtcnn/cropImage/infr_5dot.txt'
'''
im_dir = '/home/pengtao/deeplearn/mtcnn/cropImage/cmerge_32'
anno_file ='/home/pengtao/deeplearn/mtcnn/cropImage/color_5dotin.txt'
homepic='/datahome/facenet/D2_nir_color_samples/real'  #color
ourcameraflag = 0
'''
'''
im_dir = '/home/pengtao/deeplearn/mtcnn/cropImage/gmerge_32'
anno_file ='/home/pengtao/deeplearn/mtcnn/cropImage/nir_5dotin.txt'
homepic='/datahome/facenet/D2_nir_color_samples/real'  #infr
ourcameraflag = 0
'''
'''
imgsize = 48 
im_dir = '/home/pengtao/deeplearn/mtcnn/cropImage/NIR_ALL_calib_mi_%s'%imgsize
anno_file ='/datahome/facenet/NIR_ALL_calib/NIR_labelpic_landmark.txt'
homepic='/datahome/facenet/NIR_ALL_calib/NIR_labelpic'  #infr

#im_dir = 'X:/deeplearn/mtcnn/cropImage/NIR_ALL_calib_%s'%imgsize
#anno_file ='V:/NIR_ALL_calib/NIR_labelpic_landmark.txt'
#homepic='V:/NIR_ALL_calib/NIR_labelpic'  #infr

ourcameraflag =1
'''
########



#anno_file = '/home/pengtao/data/wider_face_split/wider_face_train_bbx_gt.txt'#'X:/data/wider_face_split/wider_face_train_bbx_gt.txt'
#im_dir = '/home/pengtao/data/WIDER_train/images'
pos_save_dir = im_dir + "/positive"
part_save_dir = im_dir +"/part"
neg_save_dir = im_dir + '/negative'
save_dir = im_dir
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(pos_save_dir):
    os.mkdir(pos_save_dir)
if not os.path.exists(part_save_dir):
    os.mkdir(part_save_dir)
if not os.path.exists(neg_save_dir):
    os.mkdir(neg_save_dir)
print(save_dir)
f1 = open(os.path.join(save_dir, 'pos_%s.txt'%imgsize), 'w')  #
f2 = open(os.path.join(save_dir, 'neg_%s.txt'%imgsize), 'w')  #
f3 = open(os.path.join(save_dir, 'part_%s.txt'%imgsize), 'w')  # 
#with open(anno_file, 'r') as f:
    #annotations = f.readlines()
num =  0#len(annotations)
#print ("%d pics in total" % num)
p_idx = 0 # positive
n_idx = 0 # negative
d_idx = 0 # dont care
idx = 0
box_idx = 0
imagesize = 40
sizewidth = 480
sizeheight = 572
lastidx = 1
with open(anno_file, 'r') as f:
    while True:
        annotation = f.readline()
        if not annotation:
            break
        #celebaA  
        '''
        if idx == 0:
            num =  int(annotation)
            idx+=1
            continue
        elif idx ==1:
            idx+=1
            continue   
        '''
        idx += 1
        #if idx < 37001:
            #continue
        #lastidx  =  npr.randint(50 , 80) + idx
        #for annotation in annotations:
        annotation = annotation.strip().split()
       
        #image path
        im_path = annotation[0]
        #boxed change to float type
        #bbox = map(float, annotation[1:])
        #gt 
    
        yratio = 0.35 
        xratio = 0.6
        if ourcameraflag:
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
                continue
            #print(x0,y0,x1,y1,size)
            boxes= np.concatenate((x0,y0,x1,y1),axis = 0).reshape(-1, 4)
            #print(annotation)
            #print(boxes)
        else:
            #np._float32_ma
            boxes = np.array(annotation[1:5], dtype=np.float32).reshape(-1, 4)
            #celebA 
            boxes[:,2]=  boxes[:, 2]+boxes[:, 0]
            boxes[:, 3] = boxes[:, 3]+boxes[:, 1]
            

        #load image
        imagepath = os.path.join(homepic, im_path )
        #imagepath = im_path
        img = cv2.imread(imagepath)
        #480 *572  ourcamera
        if ourcameraflag:
            if img.shape[0]!= sizeheight or img.shape[1] != sizewidth:
                img = cv2.resize(img, (sizewidth, sizeheight), interpolation=cv2.INTER_LINEAR)

        if img is None:
            continue
        #print(imagepath)
        #idx += 1
        #if idx % 100 == 0:
            #print (idx, "images done")
            
        height, width, channel = img.shape

        neg_num = 0
        #1---->50
        while neg_num < 50 :#and idx % 4 == 1:
            #neg_num's size [40,min(width, height) / 2],min_size:40 
            minV = min(width, height,400)  if min(width,height,460)>imagesize else imagesize+1
            size = npr.randint(imagesize , minV)  #  our company camera the minmax face 80-360   this is 64 -  
            #top_left
            nx = npr.randint(0, width - size)
            ny = npr.randint(0, height - size)
            #random crop
            crop_box = np.array([nx, ny, nx + size, ny + size])
            #cal iou
            Iou = IoU(crop_box, boxes,0)
            #print('IoU',Iou)
            #print(crop_box,boxes)
            #print('Iou',Iou)
            cropped_im = img[ny : ny + size, nx : nx + size, :]
            resized_im = cv2.resize(cropped_im, (imgsize, imgsize), interpolation=cv2.INTER_LINEAR)

            if np.max(Iou) < 0.08:
                # Iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
                f2.write("negative/%s.jpg"%n_idx + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
            neg_num += 1
        #as for 正 part样本
        for box in boxes:
            # box (x_left, y_top, x_right, y_bottom)
            x1, y1, x2, y2 = box
            #gt's width
            w = x2 - x1 + 1
            #gt's height
            h = y2 - y1 + 1
            #x1, y1, w, h = box
            #x2 = x1+w+1
            #y2 = y1+h+1
            # ignore small faces
            # in case the ground truth boxes of small faces are not accurate
            if max(w, h) < 56 or x1 < 0 or y1 < 0:
                continue
            for i in range(6):
                if imgsize>=min(width/2, height/2, 460):
                    break
                size = npr.randint(imgsize, min(width/2, height/2, 460) )
                # delta_x and delta_y are offsets of (x1, y1)
                if max(-size, -x1)> w:
                    print(imagepath)
                    print('\nvalue%d %d  %d\n'%(-size,-x1,w,))
                if max(-size, -y1)> h:
                    print(imagepath)
                    print('\nvalue%d %d  %d\n'%(-size,-y1,h,))
                delta_x = npr.randint(max(-size, -x1), w)
                delta_y = npr.randint(max(-size, -y1), h)
                nx1 = int(max(0, x1 + delta_x))
                ny1 = int(max(0, y1 + delta_y))
                if nx1 + size > width or ny1 + size > height:
                    continue
                crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
                Iou = IoU(crop_box, boxes,0)
        
                cropped_im = img[ny1: ny1 + size, nx1: nx1 + size, :]
                resized_im = cv2.resize(cropped_im, (imgsize, imgsize), interpolation=cv2.INTER_LINEAR)
        
                if np.max(Iou) < 0.08:
                    # Iou with all gts must below 0.3
                    save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                    f2.write("negative/%s.jpg" % n_idx + ' 0\n')
                    cv2.imwrite(save_file, resized_im)
                    n_idx += 1        
        # generate positive examples and part faces
            if 1: #positive
                cropped_im = img[int(y1) : int(y2), int(x1) : int(x2), :]
                #resize
                resized_im = cv2.resize(cropped_im, (imgsize, imgsize), interpolation=cv2.INTER_LINEAR)
                save_file = os.path.join(pos_save_dir, "%s.jpg"%p_idx)
                f1.write("positive/%s.jpg"%p_idx + ' 1 %.2f %.2f %.2f %.2f\n'%(0, 0, 0, 0))
                cv2.imwrite(save_file, resized_im)
                p_idx += 1
            for i in range(10):
                # pos and part face size [minsize*0.8,maxsize*1.25]
                size = npr.randint(int(min(w, h) * 0.85), np.ceil(1.15 * max(w, h)))
                #print('width:',w,'height:',h)

                # delta here is the offset of box center
                delta_x = npr.randint(-w * 0.15, w * 0.15)
                delta_y = npr.randint(-h * 0.15, h * 0.15)
                #show this way: nx1 = max(x1+w/2-size/2+delta_x)
                nx1 = max(x1 + w / 2 + delta_x - size / 2, 0)
                #show this way: ny1 = max(y1+h/2-size/2+delta_y)
                ny1 = max(y1 + h / 2 + delta_y - size / 2, 0)
                nx2 = nx1 + size
                ny2 = ny1 + size

                if nx2 > width or ny2 > height:
                    continue 
                crop_box = np.array([nx1, ny1, nx2, ny2])
                #yu gt de offset
                offset_x1 = (x1 - nx1) / float(size)
                offset_y1 = (y1 - ny1) / float(size)
                offset_x2 = (x2 - nx2) / float(size)
                offset_y2 = (y2 - ny2) / float(size)
                #crop
                cropped_im = img[int(ny1) : int(ny2), int(nx1) : int(nx2), :]
                #resize
                resized_im = cv2.resize(cropped_im, (imgsize, imgsize), interpolation=cv2.INTER_LINEAR)

                box_ = box.reshape(1, -1)
                iou_v = IoU(crop_box, box_,0)
                #print('iou_u',iou_v)
                #print(crop_box,box_)
                if iou_v >= 0.6:
                    save_file = os.path.join(pos_save_dir, "%s.jpg"%p_idx)
                    f1.write("positive/%s.jpg"%p_idx + ' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1
                elif iou_v >= 0.40 :
                    save_file = os.path.join(part_save_dir, "%s.jpg"%d_idx)
                    f3.write("part/%s.jpg"%d_idx + ' -1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
            box_idx += 1
        print ("%s images done, pos: %s part: %s neg: %s"%(idx, p_idx, d_idx, n_idx))
f1.close()
f2.close()
f3.close()
#'''
