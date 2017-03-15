
from tensorflow.examples.tutorials.mnist import input_data

def load_data(dirname, one_hot=True):
  return input_data.read_data_sets(dirname, one_hot=one_hot)