import numpy as np
import numpy.random as npr
import os

data_dir = 'X:\\deeplearn\\mtcnn\\cropImage\\gmerge_32'
#data_dir = 'X:\\deeplearn\\mtcnn\\cropImage\\infr_32'
#anno_file = os.path.join(data_dir, "anno.txt")

data_dirl= ['X:\\deeplearn\\mtcnn\\cropImage\\cele_32','X:\\deeplearn\\mtcnn\\cropImage\\cmerge_32',
    'X:\\deeplearn\\mtcnn\\cropImage\\gmerge_32','X:\\deeplearn\\mtcnn\\cropImage\\gmerge_32_1',
    'X:\\deeplearn\\mtcnn\\cropImage\\wider_12']

size = 32
sizelist = [32,32,32,32,12]
ratioNum = [60000,2620,4346,74752,60000]
net = "wider_12"
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
with open(os.path.join(data_dir, 'landmark_%s_aug.txt' %(size)), 'r') as f:
    landmark = f.readlines()

dir_path = os.path.join('.', 'imglists')
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
if not os.path.exists(os.path.join(dir_path, "%s" %(net))):
    os.makedirs(os.path.join(dir_path, "%s" %(net)))
with open(os.path.join(dir_path, "%s" %(net),"train_PNet_landmark.txt" ), "w") as f_all:

    for n in range(len(data_dirl)):
      
        with open(os.path.join(data_dirl[n], 'pos_%s.txt' % ( sizelist[n])), 'r') as f:
            pos = f.readlines()

        with open(os.path.join(data_dirl[n], 'neg_%s.txt' % ( sizelist[n])), 'r') as f:
            neg = f.readlines()

        with open(os.path.join(data_dirl[n], 'part_%s.txt' % ( sizelist[n])), 'r') as f:
            part = f.readlines()

        

        nums = [len(neg), len(pos), len(part)]
        #ratio = [3, 1, 1, 2]
        #base_num = min(nums)
        #base_num = 10000
        base_num = ratioNum[n]
        print(len(neg), len(pos), len(part), len(landmark), base_num)
        if len(neg) > base_num * 3:
            neg_keep = npr.choice(len(neg), size=base_num * 3, replace=True)
        else:
            neg_keep = npr.choice(len(neg), size=len(neg), replace=True)
        pos_keep = npr.choice(len(pos), size=base_num, replace=True)
        part_keep = npr.choice(len(part), size=base_num, replace=True)

        landmark_keep = npr.choice(len(landmark), size=base_num*2, replace=True)
        print(len(neg_keep), len(pos_keep), len(part_keep),len(landmark_keep))

        if 1:
            _ , name = os.path.split(data_dirl[n])
            name = name+'/'
            for i in pos_keep:
                f_all.write(name)
                f_all.write(pos[i])
            for i in neg_keep:
                f_all.write(name)
                f_all.write(neg[i])
            for i in part_keep:
                f_all.write(name)
                f_all.write(part[i])
        else:
            for i in pos_keep:
                f_all.write(pos[i])
            for i in neg_keep:
                f_all.write(neg[i])
            for i in part_keep:
                f_all.write(part[i])

        
        '''
        if len(landmark) >  (base_num*2):
            for i in landmark_keep:
                f.write(landmark[i])
        else:
            for item in landmark:
                f.write(item)
        '''
    
