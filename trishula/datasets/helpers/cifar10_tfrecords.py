
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

import os
import sys
import tarfile
from six.moves import urllib
import tensorflow as tf

DATA_URL     = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
LABEL_SIZE   = 1
IMAGE_HEIGHT = 32
IMAGE_WIDTH  = 32
IMAGE_DEPTH  = 3
IMAGE_SIZE   = IMAGE_HEIGHT*IMAGE_WIDTH*IMAGE_DEPTH

NO_TRAIN_EXAMPLES = 50000
NO_TEST_EXAMPLES = 10000
NO_EXAMPLES = NO_TRAIN_EXAMPLES + NO_TEST_EXAMPLES

def download_and_extract(dirpath):
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dirpath, filename)
  if not tf.gfile.Exists(dirpath):
    tf.gfile.MakeDirs(dirpath)
  if not tf.gfile.Exists(filepath):
    print("Downloading dataset from ", DATA_URL)
    def _progress(count, block_size, total_size):
      num_blocks = total_size/block_size
      percentage_completed = (count/num_blocks)*100
      progress_blocks = int(percentage_completed)
      sys.stdout.write('\r|' 
                       + '='*progress_blocks 
                       + '>' 
                       + ' '*(100-progress_blocks) 
                       + '|'
                       + ' %.1f%% completed'%(percentage_completed))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(URL, filepath, _progress)
  print("Extracting ", filename, "into directory -", dirpath)
  archive = tarfile.open(filepath, 'r:gz')
  archive.extractall(dirpath)
  extracted_subdir = os.path.commonprefix(archive.getnames())
  return os.path.join(dirpath, extracted_subdir)

def filename_queue(data_dir):
  filenames = tf.train.match_filenames_once(os.path.join(data_dir, '*.bin'))
  return tf.train.string_input_producer(filenames)

def training_data_reader():
  record_size = LABEL_SIZE + IMAGE_SIZE
  return tf.FixedLengthRecordReader(record_size)

def split_image_and_label_from_record(record_bytes):
  label = tf.cast(tf.slice(record_bytes, [0], [LABEL_SIZE]), tf.int32)
  image = tf.cast(tf.slice(record_bytes, [LABEL_SIZE], [IMAGE_SIZE]), tf.int32)
  image = tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])
  image = tf.transpose(image, [1,2,0])
  return image, label

def read_record(reader, input_queue):
  _, value = reader.read(input_queue)
  record_bytes = tf.decode_raw(value, tf.uint8)
  return record_bytes

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def write_tfrecords(image, label,dirpath):
  tfrecords_dir = os.path.join(dirpath, 'cifar10-tfrecords')
  if not tf.gfile.Exists(tfrecords_dir):
    tf.gfile.MakeDirs(tfrecords_dir)
  tfrecords_filename = os.path.join(tfrecords_dir, 'cifar10.tfrecords')
  tfrecord_writer = tf.python_io.TFRecordWriter(tfrecords_filename)
  session = tf.Session()
  session.run(tf.global_variables_initializer())
  coord = tf.train.Coordinator()
  tf.train.start_queue_runners(coord=coord, sess=session)
  for i in xrange(NO_EXAMPLES):
    image_data, label_data = session.run([image, label])
    img_str = image_data.tostring()
    label_str = label_data.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(IMAGE_HEIGHT),
        'width': _int64_feature(IMAGE_WIDTH),
        'depth': _int64_feature(IMAGE_DEPTH),
        'image': _bytes_feature(img_str),
        'label': _bytes_feature(label_str)
        }))
    tfrecord_writer.write(example.SerializeToString())
  tfrecord_writer.close()

def generate_cifar10_TFRecords(dirpath):
  data_dir = download_and_extract(dirpath)
  input_queue = filename_queue(data_dir)
  reader = training_data_reader()
  record_bytes = read_record(reader, input_queue)
  image, label = split_image_and_label_from_record(record_bytes)
  write_tfrecords(image, label, dirpath)

def main(_):
  generate_cifar10_TFRecords('../../../data/')

if __name__ == "__main__":
  tf.app.run()