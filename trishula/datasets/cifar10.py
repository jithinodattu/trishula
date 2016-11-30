
import os
import sys
import tarfile
import tensorflow as tf
from six.moves import urllib

from trishula.abstracts import Dataset

URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000

class Partition(object):
	def __init__(self, part_name):
		self.name = part_name

class Cifar10(Dataset):

	def __init__(self):
		self.train = Partition("train")
		self.validation = Partition("validation")
		self.test = Partition("test")

	def read_data_sets(self, dirname, one_hot):
		pass


def download(url, filename, filepath):
	def _progress(count, block_size, total_size):
		sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
					float(count * block_size) / float(total_size) * 100.0))
		sys.stdout.flush()
	filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
	statinfo = os.stat(filepath)
	print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

def read_cifar10(filename_queue):
	class CIFAR10Record(object):
		pass
	result = CIFAR10Record()

	label_bytes = 1
	result.height = 32
	result.width = 32
	result.depth = 3
	image_bytes = result.height * result.width * result.depth

	record_bytes = label_bytes + image_bytes

	reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
	result.key, value = reader.read(filename_queue)

	record_bytes = tf.decode_raw(value, tf.uint8)

	result.label = tf.cast(
	tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

	result.label = tf.one_hot(result.label, 10, 1, 0, -1)

	depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
	[result.depth, result.height, result.width])

	result.uint8image = tf.transpose(depth_major, [1, 2, 0])

	return result

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
	num_preprocess_threads = 16
	if shuffle:
		images, label_batch = tf.train.shuffle_batch(
									[image, label],
									batch_size=batch_size,
									num_threads=num_preprocess_threads,
									capacity=min_queue_examples + 3 * batch_size,
									min_after_dequeue=min_queue_examples
									)
	else:
		images, label_batch = tf.train.batch(
									[image, label],
									batch_size=batch_size,
									num_threads=num_preprocess_threads,
									capacity=min_queue_examples + 3 * batch_size
									)

	return images, tf.reshape(label_batch, [batch_size, 10])

def load_data(dirname, one_hot=True):
	if not os.path.exists(dirname):
		os.makedirs(dirname)
	filename = URL.split('/')[-1]
	filepath = os.path.join(dirname, filename)
	if not os.path.exists(filepath):
		download(URL, filename, filepath)
	tarfile.open(filepath, 'r:gz').extractall(dirname)
	data_dir = os.path.join(dirname, 'cifar-10-batches-bin')

	fileblob = os.path.join(data_dir, 'data_batch_*.bin')
	filenames = tf.train.match_filenames_once(fileblob)

	filename_queue = tf.train.string_input_producer(filenames)

	read_input = read_cifar10(filename_queue)
	float32_image = tf.cast(read_input.uint8image, tf.float32)

	distorted_image = tf.image.random_flip_left_right(float32_image)

	distorted_image = tf.image.random_brightness(distorted_image,
	                               						max_delta=63)
	distorted_image = tf.image.random_contrast(distorted_image,
	                             						lower=0.2, upper=1.8)

	std_image = tf.image.per_image_standardization(distorted_image)

	min_fraction_of_examples_in_queue = 0.4
	min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
	print ('Filling queue with %d CIFAR images before starting to train. '
	     				'This will take a few minutes.' % min_queue_examples)

	def next_batch(batch_size):
		return _generate_image_and_label_batch(distorted_image, read_input.label,
	                                     min_queue_examples, batch_size,
	                                     shuffle=True)
	cifar10 = Cifar10()
	cifar10.train.next_batch = next_batch

	return cifar10
