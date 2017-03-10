
"""
-- A helper for generating CIFAR 10 TFRecords --

At first we downloads the cifar-10 tar file from the DATA_URL as given below.
The binary files containing datasets for training and test are extracted and 
those filenames are added to string_input_producer from where the record_reader 
consumes. 

Record size is calculated based on the following assumptions
image_size = 32x32x3
label_size = 1
total_zize of record = image_size + label_size

Records are then write back to disk as TFRecords with features
  - height
  - width
  - depth
  - image
  - label

"""
from __future__ import print_function
from __future__ import division

import os
import sys
import tarfile
import glob
import logging
import numpy as np
import tensorflow as tf
from six.moves import urllib

DATA_URL     = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
LABEL_SIZE   = 1
IMAGE_HEIGHT = 32
IMAGE_WIDTH  = 32
IMAGE_DEPTH  = 3
IMAGE_SIZE   = IMAGE_HEIGHT*IMAGE_WIDTH*IMAGE_DEPTH
SAMPLE_SIZE  = IMAGE_SIZE + LABEL_SIZE
SAMPLES_PER_BIN_FILE = 10000

TFRECORDS_FILENAME = 'cifar10.tfrecords'

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger('TRISHULA')

def download_and_extract(dirpath):
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dirpath, filename)
  if not tf.gfile.Exists(dirpath):
    tf.gfile.MakeDirs(dirpath)
  if not tf.gfile.Exists(filepath):
    LOGGER.info("Downloading dataset from %s"%DATA_URL)
    def _progress(count, block_size, total_size):
      num_blocks = total_size/block_size
      percentage_completed = (count/num_blocks)*100
      sys.stdout.write('\rDownloading : %.1f%% completed'%(percentage_completed))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
  LOGGER.info("\rExtracting %s into directory - %s"%(filename, dirpath))
  archive = tarfile.open(filepath, 'r:gz')
  archive.extractall(dirpath)
  extracted_subdir = os.path.commonprefix(archive.getnames())
  return os.path.join(dirpath, extracted_subdir)

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def read_binarydata_filenames(data_dir, name_pattern):
  return glob.glob(os.path.join(data_dir, name_pattern))

def split_format_image_and_label(record):
  uint8_label = record[:1]
  uint8_image = record[1:]
  label = uint8_label.astype(np.int64)
  int64_image = uint8_image.astype(np.int64)
  image = np.reshape(int64_image, [IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH])
  image = np.transpose(image, [1, 2, 0])
  return image, label

def create_example(image, label):
  image_str = image.tostring()
  label_str = label.tostring()
  return tf.train.Example(features=tf.train.Features(feature={
    'height': _int64_feature(IMAGE_HEIGHT),
    'width': _int64_feature(IMAGE_WIDTH),
    'depth': _int64_feature(IMAGE_DEPTH),
    'image': _bytes_feature(image_str),
    'label': _bytes_feature(label_str)
    }))

def write_tfrecords(binary_filenames, dirpath):
  tfrecord_path = os.path.join(dirpath, TFRECORDS_FILENAME)
  tfrecord_writer = tf.python_io.TFRecordWriter(tfrecord_path)
  for filename in binary_filenames:
    binarystream = open(filename, 'rb')
    raw_data = binarystream.read()
    np_data = np.frombuffer(raw_data, np.uint8)
    uint8_records = np.reshape(np_data, [SAMPLES_PER_BIN_FILE, SAMPLE_SIZE])
    for record in uint8_records:
      image, label = split_format_image_and_label(record)
      example = create_example(image, label)
      tfrecord_writer.write(example.SerializeToString())
  tfrecord_writer.close()

def partition_and_save(name, data_dir, filename_pattern, destination_dir):
  records_dir = os.path.join(destination_dir, name)
  tf.gfile.MakeDirs(records_dir)
  binaryfiles = read_binarydata_filenames(data_dir, filename_pattern)
  LOGGER.info("\rWriting %s tfrecords into %s"%(name, records_dir))
  write_tfrecords(binaryfiles, records_dir)

def generate_cifar10_TFRecords(destination_dir, download_dir='/tmp'):
  extracted_dir = download_and_extract(download_dir)
  #generate train set
  partition_and_save('train', extracted_dir, 'data*[1-4].bin', destination_dir)
  #generate dev set
  partition_and_save('dev', extracted_dir, 'data*5.bin', destination_dir)
  #generate test set
  partition_and_save('test', extracted_dir, 'test*.bin', destination_dir)