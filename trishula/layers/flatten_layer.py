
import numpy as np
import tensorflow as tf
from trishula.abstracts.layer import Layer

class FlattenLayer(Layer):

  def __init__(self, order=0):
    self.order = order

  def feedforward(self, input):
    flat_tensor_dim =  np.prod(input.get_shape().as_list()[1:])
    return tf.reshape(input, [-1, flat_tensor_dim])