
""" 
-- Abstract class for creating dataset -- 

Properties
  - train
  - dev
  - test

Dataset is expected to be called as 
  - dataset.train.next_batch()
  - dataset.dev.next_batch()
  - dataset.test.next_batch()
these methods should return graph for fetching batch
"""

from abc import ABCMeta, abstractproperty, abstractmethod

class Dataset(object):
  __metaclass__ = ABCMeta

  def __init__(self, name):
    pass

  @abstractmethod
  def num_classes(self):
    pass

  @abstractproperty
  def train(self):
    pass

  @abstractproperty
  def dev(self):
    pass

  @abstractproperty
  def test(self):
    pass