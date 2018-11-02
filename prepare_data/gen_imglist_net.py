import numpy as np
import numpy.random as npr
import os

#data_dir = 'X:\\deeplearn\\mtcnn\\cropImage\\gmerge_32'
#data_dir = 'X:\\deeplearn\\mtcnn\\cropImage\\infr_32'
#anno_file = os.path.join(data_dir, "anno.txt")
'''
data_dirl= ['X:\\deeplearn\\mtcnn\\cropImage\\cele_32','X:\\deeplearn\\mtcnn\\cropImage\\cmerge_32',
    'X:\\deeplearn\\mtcnn\\cropImage\\gmerge_32','X:\\deeplearn\\mtcnn\\cropImage\\gmerge_32_1',
    'X:\\deeplearn\\mtcnn\\cropImage\\wider_12']

size = 32
sizelist = [32,32,32,32,12]
ratioNum = [1000000,50000,50000,30000,500000]
'''
data_dirl= ['X:\\deeplearn\\mtcnn\\cropImage\\CelebA_48',
    'X:\\deeplearn\\mtcnn\\cropImage\\singleNIR_ALL_calib_48',
    'X:\\deeplearn\\mtcnn\\cropImage\\singleNIR_ALL_calib_mi_48']
#data_dirl= ['X:\\deeplearn\\mtcnn\\cropImage\\NIR_ALL_calib_48','X:\\deeplearn\\mtcnn\\cropImage\\NIR_ALL_calib_mi_48']

size = 48
sizelist = [48,48,48,32,12]
ratioNum = [1000000,200000,200000,30000,500000]
markflag = 0    # 0 neg 1 pos 2 part 3 landmark
charmark = 'neg_single'


dir_path = "./imglists/OnetA"  
net = 'NIR_calib_A'
'''
if size == 12:
    net = "PNet"
elif size == 24:
    net = "RNet"
elif size == 48:
    net = "ONet"
'''
'''
with open(os.path.join(data_dir, 'pos_%s.txt' % ( size)), 'r') as f:
    pos = f.readlines()

with open(os.path.join(data_dir, 'neg_%s.txt' % ( size)), 'r') as f:
    neg = f.readlines()

with open(os.path.join(data_dir, 'part_%s.txt' % ( size)), 'r') as f:
    part = f.readlines()
'''
#with open(os.path.join(data_dir, 'landmark_%s_aug.txt' %(size)), 'r') as f:
    #landmark = f.readlines()

#dir_path = os.path.join('.', 'imglists')
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
if not os.path.exists(os.path.join(dir_path, "%s" %(net))):
    os.makedirs(os.path.join(dir_path, "%s" %(net)))


with open(os.path.join(dir_path, "%s" %(net),"train_Net_%s.txt"%charmark ), "w") as f_all:
    for n in range(len(data_dirl)):
        base_num = ratioNum[n]
        _ , name = os.path.split(data_dirl[n])
        name = name+'/'

        if markflag == 1:
            with open(os.path.join(data_dirl[n], 'pos_%s.txt' % ( sizelist[n])), 'r') as f:
                pos = f.readlines()
                print('pos',len(pos))
                pos_keep = npr.choice(len(pos), size=base_num, replace=False)
                print(len(pos_keep))
            for i in pos_keep:
                f_all.write(name)
                f_all.write(pos[i])
        elif markflag == 0:
            with open(os.path.join(data_dirl[n], 'neg_%s.txt' % ( sizelist[n])), 'r') as f:
                neg = f.readlines()
                print('neg:',len(neg))
                neg_keep = npr.choice(len(neg), size=base_num , replace=False)
                print(len(neg_keep))
            for i in neg_keep:
                f_all.write(name)
                f_all.write(neg[i])
        elif markflag == 2:
            with open(os.path.join(data_dirl[n], 'part_%s.txt' % ( sizelist[n])), 'r') as f:
                part = f.readlines()
                print('part',len(part))
                part_keep = npr.choice(len(part), size=base_num, replace=True)
                print(len(part_keep))
            for i in part_keep:
                f_all.write(name)
                f_all.write(part[i])
        elif markflag == 3:
            with open(os.path.join(data_dirl[n], 'landmark_%s_aug.txt' %(size)), 'r') as f:
                landmark = f.readlines()
                print('landmark',len(landmark))
                landmark_keep = npr.choice(len(landmark), size=base_num, replace=True)
                print(len(landmark_keep))
            for i in landmark_keep:
                f_all.write(name)
                f_all.write(landmark[i])




        
    


