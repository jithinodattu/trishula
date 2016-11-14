
import tensorflow as tf
from trishula.abstracts.optimizer import Optimizer

class GradientDescentOptimizer(Optimizer):

	def __init__(self, error, learning_rate):
		self.error = error
		self.learning_rate = learning_rate

	def generate_training_step(self, y, y_):
		return tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.error(y, y_))