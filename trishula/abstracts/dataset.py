
from abc import ABCMeta, abstractmethod

class Dataset(object):
	__metaclass__ = ABCMeta

	@abstractmethod
	def __init__(self):
		pass

	@abstractmethod
	def read_data_sets(self, dirnam, one_hot):
		pass 