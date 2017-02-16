
"""
-- Helper class for downloading & preprocessing CIFAR-10 dataset --
"""

from __future__ import print_function, division

import os
import sys
import tarfile
import tensorflow as tf
from six.moves import urllib

from trishula.abstracts import Dataset

URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

class Partition(object):
  pass

def download_and_extract(dirpath):
  filename = URL.split('/')[-1]
  filepath = os.path.join(dirpath, filename)
  if not tf.gfile.Exists(dirpath):
    tf.gfile.MakeDirs(dirpath)
  if not tf.gfile.Exists(filepath):
    print("Downloading dataset from ", URL)
    def _progress(count, block_size, total_size):
      num_blocks = total_size/block_size
      percentage_completed = (count/num_blocks)*100
      progress_blocks = int(percentage_completed)
      sys.stdout.write('\r|' 
                       + '='*progress_blocks 
                       + '>' 
                       + ' '*(100-progress_blocks) 
                       + '|'
                       + ' %.1f%% completed'%(percentage_completed))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(URL, filepath, _progress)
  print("Extracting ", filename, "into directory -", dirpath)
  archive = tarfile.open(filepath, 'r:gz')
  archive.extractall(dirpath)
  extracted_subdir = os.path.commonprefix(archive.getnames())
  return os.path.join(dirpath, extracted_subdir)

class CIFAR10(Dataset):

  def __init__(self, dirpath):
    data_dir = download_and_extract(dirpath)

  @property
  def num_classes(self):
    pass

  @property
  def train(self):
    pass

  @property
  def dev(self):
    pass

  @property
  def test(self):
    pass

cifar10 = CIFAR10('/tmp/cifar10')