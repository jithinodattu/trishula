
from tensorflow import tf 
from trishula.abstracts import Layer

class DenseLayer(Layer):

	def __init__(self, size, w_initializer=None, activ_fn=None):
		if w_initializer:
			W_value, b_value = w_initializer(size=size, activ_fn=activ_fn)
		else:
			W_value, b_value = tf.zeros(size), tf.zeros([size[1]])

		self.W = tf.Variable(W_value)
		self.b = tf.Variable(b_value)

		self.activ_fn = activ_fn

		self.params = [self.W, self.b]

	def feedforward(self, input):
		pre_activation = tf.matmul(input, self.W) + self.b
		return self.activ_fn(pre_activation) if self.activ_fn else pre_activation