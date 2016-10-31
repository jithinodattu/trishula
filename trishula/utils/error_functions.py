
import tensorflow as tf

def cross_entropy(y, y_):
	return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))