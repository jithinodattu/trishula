
import tensorflow as tf
from trishula.abstracts import Model
from trishula.abstracts import Layer

class Sequential(Model):

	def __init__(self):
		self.layers = []

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

	def optimize(self, 
		dataset, 
		error,
		learning_rate=0.50, 
		n_epochs=2000, 
		batch_size=50):

		self.X = tf.placeholder(tf.float32, shape=[None, 784], name='X')
		self.y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')

		self._connect_layers()

		error = error(self.y, self.y_)

		train_step = tf.train.AdamOptimizer(1e-4).minimize(error)

		session = tf.Session()
		session.run(tf.initialize_all_variables())

		correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		for i in range(n_epochs):
			batch_xs, batch_ys = dataset.train.next_batch(batch_size)
			if i % 100 == 0:
				train_accuracy = session.run(accuracy, feed_dict={
									self.X: batch_xs, self.y_: batch_ys})
				print("step %d, training accuracy %g" % (i, train_accuracy))
			session.run(train_step, feed_dict={self.X: batch_xs, self.y_: batch_ys})

		print("test accuracy %g" % session.run(accuracy, feed_dict={
					self.X: dataset.test.images[:50], self.y_: dataset.test.labels[:50]}))

	def predict(self):
		pass

	def save(self):
		pass

	def load(self):
		pass

