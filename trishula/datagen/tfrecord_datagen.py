
import os
import tensorflow as tf

from trishula.abstracts import DataGen

def tfrecord_name_queue(TFRecord_dirpath):
  filenames = tf.train.match_filenames_once(TFRecord_dirpath)
  return tf.train.string_input_producer(filenames)

def read_tfrecords(TFRecord_dirpath):
  filename_queue = tfrecord_name_queue(TFRecord_dirpath)
  reader = tf.TFRecordReader()
  _, example = reader.read(filename_queue)
  features = tf.parse_single_example(
                                   example, 
                                   features={
                                    'height': tf.FixedLenFeature([], tf.int64),
                                    'width': tf.FixedLenFeature([], tf.int64),
                                    'depth': tf.FixedLenFeature([], tf.int64),
                                    'image': tf.FixedLenFeature([], tf.float32),
                                    'label': tf.FixedLenFeature([], tf.int64) 
                                   })
  image = features['image']
  height = tf.cast(features['height'], tf.int32)
  width = tf.cast(features['width'], tf.int32)
  depth = tf.cast(features['depth'], tf.int32)
  label = tf.cast(features['label'], tf.int32)

  images, labels = tf.train.shuffle_batch(
                                          [image, label], 
                                          batch_size=100, 
                                          capacity=30, 
                                          num_threads=2, 
                                          min_after_dequeue=10)
  return images, labels

class DatasetContainer(object):
  pass

class Partition(object):
  pass


class TFRecordDataGen(DataGen):

  def __init__(self, TFRecord_dirpath):
    #pipeline for trainset
    train_partition = Partition()
    train_dirpth = os.path.join(TFRecord_dirpath, 'train')
    train_partition.next_batch = read_tfrecords(train_dirpth)
    #pipeline for devset
    dev_partition = Partition()
    dev_dirpth = os.path.join(TFRecord_dirpath, 'dev')
    dev_partition.next_batch = read_tfrecords(dev_dirpth)
    #pipeline for testset
    test_partition = Partition()
    test_dirpth = os.path.join(TFRecord_dirpath, 'test')
    test_partition.next_batch = read_tfrecords(test_dirpth)

    self.dataset = DatasetContainer()
    self.dataset.train = train_partition
    self.dataset.dev = dev_partition
    self.dataset.test = test_partition

  def generate_from_TFRecords(self, shuffle=True):
    return self.dataset