
import tensorflow as tf
from trishula.abstracts import Model
from trishula.abstracts import Layer

from trishula.optimizers import AdamOptimizer

class Sequential(Model):

	def __init__(self):
		self.layers = []
		self.session = tf.Session()

	def add(self, layer):
		assert issubclass(type(layer) , Layer), "This layer is not a subclass of trishula.abstracts.Layer"
		self.layers.append(layer)

	def _connect_layers(self):
		assert len(self.layers) != 0, "There is no layer in the model"
		layer_input = self.X
		for layer in self.layers:
			layer_output = layer.feedforward(layer_input)
			layer_input = layer_output
		self.y = layer_output

	def _execute(self, graph, feed_dict=None):
		return self.session.run(graph, feed_dict=feed_dict)

	def optimize(self, 
		dataset, 
		optimizer, 
		n_epochs=2000, 
		batch_size=50):

		self.X = tf.placeholder(tf.float32, name='X')
		self.y_ = tf.placeholder(tf.int32, name='y_')

		self.X, self.y_ = dataset.train.next_batch

		self._connect_layers()

		train_step = optimizer.generate_training_step(self.y, self.y_)

		self._execute(tf.global_variables_initializer())

		correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		tf.train.start_queue_runners(sess=self.session)

		for i in range(n_epochs):
			if i % 100 == 0:
				train_accuracy = self._execute(accuracy)
				print("step %d, training accuracy %g" % (i, train_accuracy))
			self._execute(train_step)
		test_accuracy = self._execute(accuracy)
		print("test accuracy %g" % test_accuracy)

	def predict(self):
		pass

	def save(self):
		pass

	def load(self):
		pass

