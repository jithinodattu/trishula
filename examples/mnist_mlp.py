
from trishula.models import Sequential
from trishula.optimizers import GradientDescentOptimizer
from trishula.layers import *

from trishula.utils import error_functions
from trishula.utils import activation_functions
from trishula.datasets import mnist

mnist_data = mnist.load_data('../data/')

relu = activation_functions.relu

model = Sequential()

model.add(
	DenseLayer(
		shape=(28*28,1024)
		)
	)

model.add(ActivationLayer(relu))

model.add(
	DenseLayer(
		shape=(1024,10)
		)
	)

model.add(SoftmaxLayer())

cross_entropy = error_functions.cross_entropy
gradient_descent_optimizer = GradientDescentOptimizer(error=cross_entropy, learning_rate=0.5)

model.optimize(
	dataset=mnist_data,
	optimizer=gradient_descent_optimizer,
	batch_size=100
	)