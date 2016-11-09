
import tensorflow as tf
from trishula.abstracts.layer import Layer

class FlattenLayer(Layer):

	def __init__(self, order=0):
		self.order = order

	def feedforward(input):
		return tf.reshape(input, [-1, input.get_shape().as_list()[self.order]])