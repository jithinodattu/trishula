
"""
-- Helper class for downloading & preprocessing CIFAR-10 dataset --
"""

from __future__ import print_function, division

import os
import sys
import tarfile
import tensorflow as tf
from six.moves import urllib

from trishula.datagen import TFRecordDataGen

def cifar10():
  cifar10_datagen = TFRecordDataGen()
  return cifar10_datagen.generate_dataset()