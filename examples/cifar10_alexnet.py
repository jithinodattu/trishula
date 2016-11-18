
from trishula.models import Sequential
from trishula.optimizers import AdamOptimizer
from trishula.layers import *

from trishula.utils import error_functions
from trishula.utils import activation_functions
from trishula.datasets import cifar10

cifar10_data = cifar10.load_data('data/')
relu = activation_functions.relu

BATCH_SIZE=100

model = Sequential()

model.add(
	Convolution2DLayer(
		filter_shape=(3,3,3, 64),
		input_shape=(BATCH_SIZE,32,32,3),
		padding='VALID',
		strides=1
		)
	)

model.add(ActivationLayer(relu))

model.add(
	Convolution2DLayer(
		filter_shape=(3,3,64,128),
		input_shape=(BATCH_SIZE,30,30,64),
		padding='VALID',
		strides=1
		)
	)

model.add(ActivationLayer(relu))

model.add(
	MaxPooling2DLayer(
		shape=2,
		strides=2
		)
	)

model.add(LocalResponseNormalizationLayer(depth_radius=4))

model.add(
	Convolution2DLayer(
		filter_shape=(5,5,128,256),
		input_shape=(BATCH_SIZE,14,14,128),
		strides=1
		)
	)

model.add(ActivationLayer(relu))

model.add(
	MaxPooling2DLayer(
		shape=2,
		strides=2
		)
	)

model.add(LocalResponseNormalizationLayer(depth_radius=4))

model.add(FlattenLayer())

model.add(
	DenseLayer(
		shape=(7*7*256,4096)
		)
	)

model.add(DropOutLayer(9.0))

model.add(ActivationLayer(relu))

model.add(
	DenseLayer(
		shape=(4096,4096)
		)
	)

model.add(DropOutLayer(9.0))

model.add(ActivationLayer(relu))

model.add(
	DenseLayer(
		shape=(4096,10)
		)
	)

softmax_cross_entropy = error_functions.softmax_cross_entropy
adam_optimizer = AdamOptimizer(error=softmax_cross_entropy, learning_rate=1e-4)

model.optimize(
	dataset=cifar10_data,
	optimizer=adam_optimizer,
	n_epochs=10000,
	batch_size=BATCH_SIZE
	)