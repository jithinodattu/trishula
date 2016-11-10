
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

def gloret(name, shape):
	return tf.get_variable(name, shape=shape, 
		initializer=xavier_initializer())

def truncated_normal(name, shape, mean=0.0, stddev=1.0):
	initial =  tf.truncated_normal(shape, mean, stddev, name=name)
	return tf.Variable(initial)

def constant(name, shape, value=0.1):
	initial = tf.constant(value, shape=shape)
	return tf.Variable(initial, name=name)

def zeros(name, shape):
	return tf.Variable(tf.zeros(shape, name=name))
