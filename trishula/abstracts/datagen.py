
"""
-- Generator class for datasets --
This helper class created batches from TFRecords
"""

from abc import ABCMeta, abstractmethod

class DataGen(object):
  __metaclass__ = ABCMeta

  def __init__(self, TFRecords_dir, dataset_name):
    pass

  @abstractmethod
  def generate_from_TFRecords(partition_rule, shuffle=True):
    pass