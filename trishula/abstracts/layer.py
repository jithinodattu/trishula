
from abc import ABCMeta, abstractmethod

class Layer(object):
	"""
	Abstract class for creating layer.

	`
	 feedforward -> Implement functionality of this layer
	`
	"""
	__metaclass__ = ABCMeta

	@abstractmethod
	def __init__(self):
		pass

	@abstractmethod
	def feedforward(self):
		pass