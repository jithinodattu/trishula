
from trishula.models import Sequential
from trishula.layers import DenseLayer, SoftmaxLayer

from trishula.utils import error_functions
from trishula.datasets import mnist

mnist_data = mnist.load_data('../data/')
input_dim  = 28*28
output_dim = 10

cross_entropy = error_functions.cross_entropy

model = Sequential()
model.add(DenseLayer(shape=[input_dim, output_dim]))
model.add(SoftmaxLayer())

model.optimize(
	dataset=mnist_data,
	error=cross_entropy,
	learning_rate=0.5
	)