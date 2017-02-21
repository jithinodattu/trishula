
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
                                    'image': tf.FixedLenFeature([], tf.string),
                                    'label': tf.FixedLenFeature([], tf.int64) 
                                   })
  image = tf.decode_raw(features['image'], tf.uint8)
  height = tf.cast(features['height'], tf.int32)
  width = tf.cast(features['width'], tf.int32)
  depth = tf.cast(features['depth'], tf.int32)
  label = tf.cast(features['label'], tf.int32)

  images, labels = tf.train.shuffle_batch(
                                          [image, label], 
                                          batch_size=10, 
                                          capacity=30, 
                                          num_threads=2, 
                                          min_after_dequeue=10)

class DatasetContainer(object):
  pass

class Partition(object):
  pass


class TFRecordDataGen(DataGen):

  def __init__(self, TFRecord_dirpath):
    training_partition = Partition()
    training_partition.next_batch = read_tfrecords(TFRecord_dirpath)
    self.dataset = DatasetContainer()
    self.dataset.train = training_partition
    self.dataset.test = training_partition

  def generate_dataset(partition_rule, shuffle=True):
    return self.dataset