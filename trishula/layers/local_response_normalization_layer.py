
import tensorflow as tf
from trishula.abstracts.layer import Layer

class LocalResponseNormalizationLayer(Layer):

  def __init__(self, depth_radius, bias=1.0, alpha=0.001/9.0, beta=0.75):
    self.depth_radius = depth_radius
    self.bias = bias
    self.alpha = alpha
    self.beta = beta

  def feedforward(self, input):
    return tf.nn.lrn(
          input, 
          bias=self.bias, 
          depth_radius=self.depth_radius, 
          alpha=self.alpha, 
          beta=self.beta
          )