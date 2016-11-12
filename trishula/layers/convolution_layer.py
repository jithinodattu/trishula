
import tensorflow as tf
from trishula.abstracts.layer import Layer
from trishula.utils import weight_initializers

class Convolution2DLayer(Layer):

	def __init__(
			self, 
			filter_shape, 
			input_shape,
			strides=1, 
			padding='SAME', 
			w_initializer=weight_initializers.truncated_normal,
			b_initializer=weight_initializers.constant):
		self.W = w_initializer(self.__class__.__name__+'_'+str(id(self))+'_W', filter_shape)
		self.b = b_initializer(self.__class__.__name__+'_'+str(id(self))+'_b', filter_shape[-1:])

		self.input_shape = [-1] + list(input_shape[1:])
		self.strides = [1, strides, strides, 1]
		self.padding = padding

		self.params = [self.W, self.b]

	def feedforward(self, input):
		input = tf.reshape(input, self.input_shape)
		pre_activation = tf.nn.conv2d(
						input, 
						self.W, 
						strides=self.strides, 
						padding=self.padding
						)
		return pre_activation + self.b
