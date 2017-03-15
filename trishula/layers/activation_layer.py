
import tensorflow as tf
from trishula.abstracts.layer import Layer
from trishula.utils.activation_functions import relu

class ActivationLayer(Layer):

  def __init__(self, activ_fn=relu):
    self.activ_fn = activ_fn

  def feedforward(self, input):
    return self.activ_fn(input)