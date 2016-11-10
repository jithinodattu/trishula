
import tensorflow as tf
from trishula.abstracts import Layer
from trishula.utils import weight_initializers

class DenseLayer(Layer):

	def __init__(
			self, 
			shape, 
			w_initializer=weight_initializers.gloret,
			b_initializer=weight_initializers.gloret, 
			activ_fn=None):
		self.W = w_initializer(self.__class__.__name__+'_'+str(id(self))+'_W', shape)
		self.b = w_initializer(self.__class__.__name__+'_'+str(id(self))+'_b', shape[-1:])
		self.activ_fn = activ_fn

		self.params = [self.W, self.b]

	def feedforward(self, input):
		pre_activation = tf.matmul(input, self.W) + self.b
		return self.activ_fn(pre_activation) if self.activ_fn else pre_activation