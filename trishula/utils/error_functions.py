
import tensorflow as tf

def cross_entropy(y, y_):
	return tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y+1e-25), reduction_indices=[1]))

def softmax_cross_entropy(y, y_):
	return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))