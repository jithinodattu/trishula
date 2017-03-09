
from trishula.models import Sequential
from trishula.optimizers import AdamOptimizer
from trishula.layers import *

from trishula import relu, softmax_cross_entropy
from trishula.datasets import cifar10

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

model.add(DropOutLayer(0.9))

model.add(ActivationLayer(relu))

model.add(
  DenseLayer(
    shape=(4096,10)
    )
  )

adam_optimizer = AdamOptimizer(error=softmax_cross_entropy, learning_rate=1e-4)

model.optimize(
  dataset=cifar10,
  optimizer=adam_optimizer,
  n_epochs=10000,
  batch_size=BATCH_SIZE
  )