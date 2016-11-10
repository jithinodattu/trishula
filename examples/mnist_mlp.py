
from trishula.models import Sequential
from trishula.layers import *

from trishula.utils import error_functions
from trishula.utils import activation_functions
from trishula.datasets import mnist

mnist_data = mnist.load_data('../data/')

cross_entropy = error_functions.cross_entropy
relu = activation_functions.relu

model = Sequential()

model.add(
	DenseLayer(
		shape=(28*28,1024), 
		activ_fn=relu
		)
	)

model.add(
	DenseLayer(
		shape=(1024,10)
		)
	)

model.add(SoftmaxLayer())

model.optimize(
	dataset=mnist_data,
	error=cross_entropy,
	learning_rate=0.5,
	batch_size=100
	)