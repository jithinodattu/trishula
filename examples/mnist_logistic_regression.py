
from trishula.models import Sequential
from trishula.optimizers import GradientDescentOptimizer
from trishula.layers import *

from trishula.utils import error_functions
from trishula.utils import weight_initializers
from trishula.datasets import mnist

mnist_data = mnist.load_data('../data/')
input_dim  = 28*28
output_dim = 10

model = Sequential()

model.add(
	DenseLayer(
		shape=[input_dim, output_dim],
		w_initializer=weight_initializers.zeros,
		b_initializer=weight_initializers.zeros
		)
	)

model.add(SoftmaxLayer())

cross_entropy = error_functions.cross_entropy
gradient_descent_optimizer = GradientDescentOptimizer(error=cross_entropy, learning_rate=0.5)

model.optimize(
	dataset=mnist_data,
	optimizer=gradient_descent_optimizer
	)