
from abc import ABCMeta, abstractmethod

class Model(object):
	"""
	Abstract class for creating model. 

	`optimize -> Does the backprop and weight optimization
	 predict  -> Predicts for given input
	 save 	  -> Saves the trained weights
	 load	  -> Loads weights from pretrained models
	`
	"""
	__metaclass__ = ABCMeta

	def __init__(self):
		pass

	@abstractmethod
	def optimize(self):
		pass

	@abstractmethod
	def predict(self):
		pass

	@abstractmethod
	def save(self):
		pass

	@abstractmethod
	def load(self):
		pass