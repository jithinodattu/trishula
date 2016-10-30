
from tensorflow.examples.tutorials.mnist import input_data

def load_data(dir):
	return input_data.read_data_sets("dir", one_hot=True)