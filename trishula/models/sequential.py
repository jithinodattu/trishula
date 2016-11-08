
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
		n_epochs=1000, 
		batch_size=100):

		self.X = tf.placeholder(tf.float32, name='X')
		self.y_ = tf.placeholder(tf.float32, name='y_')

		self._connect_layers()

		error = error(self.y, self.y_)

		train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)

		session = tf.InteractiveSession()
		tf.initialize_all_variables().run()

		for _ in range(n_epochs):
			batch_xs, batch_ys = dataset.train.next_batch(batch_size)
			session.run(train_step, feed_dict={self.X: batch_xs, self.y_: batch_ys})

		correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))

		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		print(session.run(accuracy, feed_dict={self.X: dataset.test.images,
		    							self.y_: dataset.test.labels}))

	def predict(self):
		pass

	def save(self):
		pass

	def load(self):
		pass

