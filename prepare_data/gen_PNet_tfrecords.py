#coding:utf-8
import os
import random
import sys
import time

import tensorflow as tf
#sys.path.append('X:\\deeplearn\\mtcnn\\MTCNN-Tensorflow-master_change\\prepare_data')
from tfrecord_utils import _process_image_withoutcoder, _convert_to_example_simple

#imagepath ='X:\\deeplearn\\mtcnn\\MTCNN_TF'
#imagepath ='X:\\deeplearn\\mtcnn\\MTCNN-Tensorflow-master_change'
#imagepath ="/home/pengtao/deeplearn/mtcnn/MTCNN-Tensorflow-master_change"
#imagepath = 'X:/deeplearn/mtcnn/cropImage/color_32'
#imagepath = '/home/pengtao/deeplearn/mtcnn/cropImage/color_32'
imagepath = '/home/pengtao/deeplearn/mtcnn/cropImage/NIR_ALL_calib_mi_48'#/cmerge_32'
mainpath = './imglists/PnetA/NIR_calib_mi'
#readTxt = mainpath + '/part_train_PNet.txt' 
modlename = 'part'
#readTxt = './imglists/PnetA/NIR_calib_mi/train_PNet_neg.txt'
readTxt = '/home/pengtao/deeplearn/mtcnn/cropImage/NIR_ALL_calib_mi_48/%s_48.txt'%modlename
#readTxt= '/home/pengtao/deeplearn/mtcnn/cropImage/cmerge_32/part_32.txt'
COLOR_GRAY = 1  # color 0  gray 1   stop -1


if COLOR_GRAY == 0 or COLOR_GRAY == 3 :
    outPathName = mainpath+'/PNet_12_color_part.tfrecord'  # color
elif COLOR_GRAY ==1:
    outPathName = mainpath+'/PNet_12_gray_%s.tfrecord'%modlename  #gray

def _add_to_tfrecord(filename, image_example, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.

    Args:
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    print('---', filename)
    #imaga_data:array to string
    #height:original image's height
    #width:original image's width
    #image_example dict contains image's info

    image_data, height, width = _process_image_withoutcoder(filename,COLOR_GRAY,12)
    example = _convert_to_example_simple(image_example, image_data)
    tfrecord_writer.write(example.SerializeToString())

'''
def _get_output_filename(output_dir, name, net):
    #st = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    #return '%s/%s_%s_%s.tfrecord' % (output_dir, name, net, st)
    return '%s/train_PNet_12_landmarkmerge.tfrecord' % (output_dir)
'''  

def run( dataset_dir, net, output_dir, name='MTCNN',shuffling=False):
    """Runs the conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    
    #tfrecord name 
    tf_filename = outPathName   # _get_output_filename(output_dir, name, net)
    if tf.gfile.Exists(tf_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return
    # GET Dataset, and shuffling.
  
    dataset = get_dataset(dataset_dir, net=net)
    # filenames = dataset['filename']
    if shuffling:
        tf_filename = tf_filename + '_shuffle'
        print(tf_filename)
        #andom.seed(12345454)
        random.shuffle(dataset)
    # Process dataset files.
    # write the data to tfrecord
    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        for i, image_example in enumerate(dataset):
            sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(dataset)))
            sys.stdout.flush()
            filename = image_example['filename']

            filename = os.path.join(imagepath, filename)
            if  filename.find('.jpg') == -1:
                filename = filename +'.jpg'
            _add_to_tfrecord(filename, image_example, tfrecord_writer)
    # Finally, write the labels file:
    # labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    print('\nFinished converting the MTCNN dataset!')


def get_dataset(dir, net='PNet'):
    #item = 'imglists/PNet/train_%s_raw.txt' % net
    item =  readTxt#'./imglists/PNet_merge/train_%s_landmarkmerge.txt' % net
    
    dataset_dir = item #os.path.join(dir, item)
    imagelist = open(dataset_dir, 'r')

    dataset = []
    for line in imagelist.readlines():
        info = line.strip().split(' ')
        data_example = dict()
        bbox = dict()
        data_example['filename'] = info[0]
        data_example['label'] = int(info[1])
        bbox['xmin'] = 0
        bbox['ymin'] = 0
        bbox['xmax'] = 0
        bbox['ymax'] = 0
        
        bbox['xlefteye'] = 0
        bbox['ylefteye'] = 0
        bbox['xrighteye'] = 0
        bbox['yrighteye'] = 0
        bbox['xnose'] = 0
        bbox['ynose'] = 0
        bbox['xleftmouth'] = 0
        bbox['yleftmouth'] = 0
        bbox['xrightmouth'] = 0
        bbox['yrightmouth'] = 0        
        if len(info) == 6:
            bbox['xmin'] = float(info[2])
            bbox['ymin'] = float(info[3])
            bbox['xmax'] = float(info[4])
            bbox['ymax'] = float(info[5])
        if len(info) == 12:
            bbox['xlefteye'] = float(info[2])
            bbox['ylefteye'] = float(info[3])
            bbox['xrighteye'] = float(info[4])
            bbox['yrighteye'] = float(info[5])
            bbox['xnose'] = float(info[6])
            bbox['ynose'] = float(info[7])
            bbox['xleftmouth'] = float(info[8])
            bbox['yleftmouth'] = float(info[9])
            bbox['xrightmouth'] = float(info[10])
            bbox['yrightmouth'] = float(info[11])
        
        data_example['bbox'] = bbox
        dataset.append(data_example)

    return dataset


if __name__ == '__main__':
    dir = '.' 
    net = 'RNet'
    output_directory = './imglists/RNet'
    run(dir, net, output_directory, shuffling=True)
