
from abc import ABCMeta, abstractmethod

class Optimizer(object):
	__metaclass__ = ABCMeta

	def __init__(self):
		pass

	@abstractmethod
	def generate_training_step(self):
		pass
