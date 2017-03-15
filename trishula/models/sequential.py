
import time
import tensorflow as tf
from datetime import datetime
from trishula.abstracts import Model
from trishula.abstracts import Layer

from trishula.optimizers import AdamOptimizer

class Sequential(Model):

  def __init__(self):
    self.layers = []
    self.session = tf.Session()

  def add(self, layer):
    assert issubclass(type(layer) , Layer), "This layer is not a subclass of trishula.abstracts.Layer"
    self.layers.append(layer)

  def _connect_layers(self):
    assert len(self.layers) != 0, "There is no layer in the model"
    layer_input = self.X
    for layer in self.layers:
      layer_output = layer.feedforward(layer_input)
      layer_input = layer_output
    self.y = layer_output

  def _execute(self, graph, feed_dict=None):
    return self.session.run(graph, feed_dict=feed_dict)

  def optimize(self, 
    dataset, 
    optimizer, 
    checkpoint_dir,
    n_epochs=2000, 
    batch_size=50):

    self.X = tf.placeholder(tf.float32, name='X')
    self.y_ = tf.placeholder(tf.int32, name='y_')

    self.X, self.y_ = dataset.train.next_batch

    self._connect_layers()

    loss = optimizer.loss(self.y, self.y_)
    train_step = optimizer.generate_training_step(self.y, self.y_)

    correct_prediction = tf.equal(tf.argmax(self.y, 1), self.y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    global_step = tf.contrib.framework.get_or_create_global_step()

    class _LoggerHook(tf.train.SessionRunHook):

      def begin(self):
        self._step = -1

      def before_run(self, run_context):
        self._step += 1
        self._start_time = time.time()
        return tf.train.SessionRunArgs(loss)

      def after_run(self, run_context, run_values):
        duration = time.time() - self._start_time
        loss_value = run_values.results
        if self._step % 10 == 0:
          
          num_examples_per_step = batch_size
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = float(duration)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=checkpoint_dir,
        hooks=[tf.train.StopAtStepHook(last_step=n_epochs), _LoggerHook()],
        config=tf.ConfigProto(log_device_placement=False)) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_step)

  def predict(self):
    pass

  def save(self):
    pass

  def load(self):
    pass

