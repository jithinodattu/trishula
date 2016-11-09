
import tensorflow as tf
from trishula.abstracts import Model
from trishula.abstracts import Layer 

class Sequential(Model):

	def __init__(self, name):
		self.name = name
		self.layers = []

	def add(self, layer):
		assert issubclass(type(layer), Layer), 
			+ "This layer is not a subclass of trishula.abstracts.Layer"
		layers.add(layer)

	def optimize(self, input_dim, output_dim):
		X = tf.placeholder(tf.float32, shape=input_dim)
		y_ = tf.placeholder(tf.float32, shape=output_dim)
		y = tf.matmul(x,self.layers[0].W) + self.layers[0].b

		sess = tf.InteractiveSession()
		sess.run(tf.initialize_all_variables())
		train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
		correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		for i in range(1000):
			batch = mnist.train.next_batch(100)
			train_step.run(feed_dict={x: batch[0], y_: batch[1]})
			if i%100==0:
				print("Accuracy after iter : ", i, accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

	def predict(self):
		pass

	def save(self):
		pass

	def load(self):
		pass

