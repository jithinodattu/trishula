
import tensorflow as tf
from trishula.abstracts.layer import Layer
from trishula.utils import weight_initializers

class MaxPooling2DLayer(Layer):

	def __init__(self, shape, strides=None, padding='SAME'):
		self.shape = [1, shape, shape, 1]
		if not strides:
			strides = shape
		self.strides = [1, strides, strides, 1]
		self.padding = padding

	def feedforward(self, input):
		return tf.nn.max_pool(
						input, 
						ksize=self.shape, 
						strides=self.strides, 
						padding=self.padding
						)