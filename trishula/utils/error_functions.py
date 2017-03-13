
import tensorflow as tf

def cross_entropy(y, y_):
	return tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y+1e-25), reduction_indices=[1]))

def softmax_cross_entropy(y, y_):
	return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

def sparse_softmax_cross_entropy(y, y_):
	cross_entropy_mean = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_))
	tf.add_to_collection('losses', cross_entropy_mean)
	return tf.add_n(tf.get_collection('losses'), name='total_loss')