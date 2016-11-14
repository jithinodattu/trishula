
import tensorflow as tf
from six.moves import urllib

from trishula.abstracts import Dataset

URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

class Cifar10(Dataset):

	def __init__(self):
		pass

	def read_data_sets(self, dirnam, one_hot):


def download():
	def _progress(count, block_size, total_size):
		sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
					float(count * block_size) / float(total_size) * 100.0))
		sys.stdout.flush()
	filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
	statinfo = os.stat(filepath)
	print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
	tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def load_data(dirname, one_hot=True):
	if not os.path.exists(dirname):
		os.makedirs(dirname)
	filename = URL.split('/')[-1]
	filepath = os.path.join(dest_directory, filename)
	if not os.path.exists(filepath):
		download()