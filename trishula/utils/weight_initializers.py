
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

def gloret(name, shape):
	return tf.get_variable(name, shape=shape, 
		initializer=xavier_initializer())
