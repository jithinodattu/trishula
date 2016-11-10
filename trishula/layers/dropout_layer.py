
import tensorflow as tf
from trishula.abstracts.layer import Layer

class DropOutLayer(Layer):

	def __init__(self, keep_prob):
		self.keep_prob = tf.Variable(keep_prob)

	def feedforward(self, input):
		return tf.nn.dropout(input, self.keep_prob)