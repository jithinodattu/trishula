
import tensorflow as tf
from trishula.abstracts import Layer
from trishula.utils.weight_initializer import gloret

class DenseLayer(Layer):

	def __init__(self, shape, w_initializer=gloret, activ_fn=None):
		self.W = gloret('W', shape)
		self.b = gloret('b', shape[-1])

		self.activ_fn = activ_fn

		self.params = [self.W, self.b]

	def feedforward(self, input):
		pre_activation = tf.matmul(input, self.W) + self.b
		return self.activ_fn(pre_activation) if self.activ_fn else pre_activation