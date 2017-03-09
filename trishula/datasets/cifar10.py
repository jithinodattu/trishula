
"""
-- Helper class for downloading & preprocessing CIFAR-10 dataset --
"""
import os
from trishula.datagen import TFRecordDataGen
from trishula.datasets.helpers import cifar10_tfrecords

SUBDIRS = ['dev', 'test', 'train']
CIFAR10_TFRECORDS_DIR = '/tmp/cifar10'

def _generate():
  if not os.path.exists(CIFAR10_TFRECORDS_DIR) or \
          sorted(os.listdir(CIFAR10_TFRECORDS_DIR)) != SUBDIRS:
    cifar10_tfrecords.generate_cifar10_TFRecords(CIFAR10_TFRECORDS_DIR)
  cifar10_datagen = TFRecordDataGen(CIFAR10_TFRECORDS_DIR)
  return cifar10_datagen.generate_from_TFRecords()

cifar10 = _generate()