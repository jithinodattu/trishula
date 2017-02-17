
"""
-- Helper class for downloading & preprocessing CIFAR-10 dataset --
"""

from __future__ import print_function, division

import os
import sys
import tarfile
import tensorflow as tf
from six.moves import urllib

from trishula.abstracts import Dataset

DATA_URL     = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
LABEL_SIZE   = 1
IMAGE_HEIGHT = 32
IMAGE_WIDTH  = 32
IMAGE_DEPTH  = 3
IMAGE_SIZE   = IMAGE_HEIGHT*IMAGE_WIDTH*IMAGE_DEPTH

class Partition(object):
  pass

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
  filenames = [ os.path.join(data_dir, 'data_batch_%d.bin'%i) for i in xrange(1,6) ]
  return tf.train.string_input_producer(filenames)

def training_data_reader():
  record_size = LABEL_SIZE + IMAGE_SIZE
  return tf.FixedLengthRecordReader(record_size)

def split_image_and_label_from_record(record_bytes):
  label = tf.cast(tf.slice(record_bytes, [0], [LABEL_SIZE]), tf.int32)
  image = tf.cast(tf.slice(record_bytes, [LABEL_SIZE], [IMAGE_SIZE]), tf.int32)
  image = tf.transpose(image, [1,2,0])
  return image, label

def read_record(reader, input_queue):
  _, value = reader.read(input_queue)
  record_bytes = tf.decode_raw(value, tf.uint8)

def generate_batch(image, label, shuffle=False):
  num_preprocess_threads = 16
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

class CIFAR10(Dataset):

  def __init__(self, dirpath):
    data_dir = download_and_extract(dirpath)
    input_queue = filename_queue(data_dir)
    reader = training_data_reader()
    record_bytes = read_record(reader, input_queue)
    image, label = split_image_and_label_from_record(record_bytes)



  @property
  def num_classes(self):
    pass

  @property
  def train(self):
    pass

  @property
  def dev(self):
    pass

  @property
  def test(self):
    pass

cifar10 = CIFAR10('/tmp/cifar10')