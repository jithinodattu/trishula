
from trishula.models import Sequential
from trishula.optimizers import AdamOptimizer
from trishula.layers import *

from trishula.utils import error_functions
from trishula.utils import activation_functions
from trishula.datasets import mnist

mnist_data = mnist.load_data('../data/')

relu = activation_functions.relu

n_kerns= [32, 64]
batch_size=50

model = Sequential()

model.add(
	Convolution2DLayer(
		filter_shape=(5,5,1,n_kerns[0]),
		input_shape=(batch_size,28,28,1),
		strides=1
		)
	)

model.add(
	MaxPooling2DLayer(
		shape=2,
		strides=2
		)
	)

model.add(ActivationLayer(relu))

model.add(
	Convolution2DLayer(
		filter_shape=(5,5,n_kerns[0],n_kerns[1]),
		input_shape=(batch_size,14,14,n_kerns[0]),
		strides=1
		)
	)

model.add(
	MaxPooling2DLayer(
		shape=2,
		strides=2
		)
	)

model.add(ActivationLayer(relu))

model.add(FlattenLayer())

model.add(
	DenseLayer(
		shape=(7*7*64,1024)
		)
	)

model.add(DropOutLayer(9.0))

model.add(ActivationLayer(relu))

model.add(
	DenseLayer(
		shape=(1024, 10)
		)
	)

model.add(SoftmaxLayer())

cross_entropy = error_functions.cross_entropy
adam_optimizer = AdamOptimizer(error=cross_entropy, learning_rate=1e-4)

model.optimize(
	dataset=mnist_data,
	optimizer=adam_optimizer,
	batch_size=batch_size
	)