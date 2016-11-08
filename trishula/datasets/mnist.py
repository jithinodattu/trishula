
from tensorflow.examples.tutorials.mnist import input_data

def load_data(dir, one_hot=True):
	return input_data.read_data_sets(dir, one_hot)