
import tensorflow as tf
from trishula.abstracts.layer import Layer
from trishula.utils import weight_initializers

class Convolution2DLayer(Layer):

	def __init__(self, shape, strides=1, padding='SAME', w_initializer=weight_initializers.gloret, activ_fn=None):
		self.W = w_initializer('W', shape)
		self.b = w_initializer('b', shape[-1])

		self.activ_fn = activ_fn
		self.strides = [1, strides, strides, 1]
		self.padding = padding

		self.params = [self.W, self.b]

	def feedforward(self, input):
		pre_activation = tf.nn.conv2d(
						input, 
						self.W, 
						strides=self.strides, 
						padding=self.padding
						)

		pre_activation = tf.nn.bias_add(pre_activation, self.b)
		return self.activ_fn(pre_activation) if self.activ_fn else pre_activation
