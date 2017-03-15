
import tensorflow as tf
from trishula.abstracts.layer import Layer

class SoftmaxLayer(Layer):

  def __init__(self, name='softmax'):
    self.name = name

  def feedforward(self, input):
    return tf.nn.softmax(input)