
import tensorflow as tf
from trishula.abstracts import Layer
from trishula.utils import weight_initializers

class DenseLayer(Layer):

	def __init__(
			self, 
			shape, 
			w_initializer=weight_initializers.gloret,
			b_initializer=weight_initializers.gloret):
		self.W = w_initializer(self.__class__.__name__+'_'+str(id(self))+'_W', shape)
		self.b = b_initializer(self.__class__.__name__+'_'+str(id(self))+'_b', shape[-1:])

		self.params = [self.W, self.b]

	def feedforward(self, input):
		pre_activation = tf.matmul(input, self.W) + self.b
		return pre_activation