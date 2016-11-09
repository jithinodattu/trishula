
import tensorflow as tf
from trishula.abstracts.layer import Layer
from trishula.utils import weight_initializers

class MaxPooling2DLayer(Layer):

	def __init__(self, shape=[2,2], strides=2, padding='SAME'):
		self.shape = [1] + shape + [1]
		self.strides = [1, strides, strides, 1]
		self.padding = padding

	def feedforward(self, input):
		return tf.nn.max_pool(input, self.shape, self.strides, self.padding)