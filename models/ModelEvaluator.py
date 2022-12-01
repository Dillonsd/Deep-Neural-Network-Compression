import os
import time
import numpy as np
import tensorflow as tf

class ModelEvaluator:
  def __init__(self, setup_model, x_test, y_test, output_path, device, eval_batch_size=1000):
    self.setup_model_func = setup_model
    self.x_test = x_test
    self.y_test = y_test
    self.output_path = output_path
    self.evaluation_batch_size = eval_batch_size
    self.device = device
    self.baseline_results = {}
    self.quantized_results = {}
    self.pruned_results = {}
    self.clustered_results = {}

  def _evaluate_baseline(self):
    model = self.setup_model_func()
    model.load_weights(os.path.join(os.path.abspath(self.output_path), 'baseline_model.h5'))
    predictions = []
    times = []
    with tf.device(self.device):
      for img in self.x_test[0:self.evaluation_batch_size]:
        start_time = time.time()
        output = model.predict(np.expand_dims(img, axis=0), verbose=0)
        times.append(time.time() - start_time)
        predictions.append(np.argmax(output[0]))
    predictions = np.array(predictions)
    self.baseline_results['accuracy'] = sum(predictions == self.y_test[0:self.evaluation_batch_size]) \
                                          / len(predictions) * 100
    self.baseline_results['latency'] = sum(times) * 1000 / len(times)
    self.baseline_results['size'] = os.path.getsize(os.path.join(os.path.abspath(self.output_path),
      'baseline_model.h5')) / 1e6
  
  def _evaluate_quantized(self):
    interpreter = tf.lite.Interpreter(model_path=os.path.join(os.path.abspath(self.output_path),
      'quantized_model.tflite'))
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    predictions = []
    times = []
    with tf.device(self.device):
      for img in self.x_test[0:self.evaluation_batch_size]:
        start_time = time.time()
        interpreter.set_tensor(input_index, np.expand_dims(img, axis=0).astype(np.float32))
        interpreter.invoke()
        times.append(time.time() - start_time)
        predictions.append(np.argmax(interpreter.tensor(output_index)()[0]))
    predictions = np.array(predictions)
    self.quantized_results['accuracy'] = sum(predictions == self.y_test[0:self.evaluation_batch_size]) \
                                          / len(predictions) * 100
    self.quantized_results['latency'] = sum(times) * 1000 / len(times)
    self.quantized_results['size'] = os.path.getsize(os.path.join(os.path.abspath(self.output_path),
      'quantized_model.tflite')) / 1e6
  
  def _evaluate_pruned(self):
    interpreter = tf.lite.Interpreter(model_path=os.path.join(os.path.abspath(self.output_path),
      'pruned_model.tflite'))
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    predictions = []
    times = []
    with tf.device(self.device):
      for img in self.x_test[0:self.evaluation_batch_size]:
        start_time = time.time()
        interpreter.set_tensor(input_index, np.expand_dims(img, axis=0).astype(np.float32))
        interpreter.invoke()
        times.append(time.time() - start_time)
        predictions.append(np.argmax(interpreter.tensor(output_index)()[0]))
    predictions = np.array(predictions)
    self.pruned_results['accuracy'] = sum(predictions == self.y_test[0:self.evaluation_batch_size]) \
                                          / len(predictions) * 100
    self.pruned_results['latency'] = sum(times) * 1000 / len(times)
    self.pruned_results['size'] = os.path.getsize(os.path.join(os.path.abspath(self.output_path),
      'pruned_model.tflite')) / 1e6
  
  def _evaluate_clustered(self):
    interpreter = tf.lite.Interpreter(model_path=os.path.join(os.path.abspath(self.output_path),
      'clustered_model.tflite'))
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    predictions = []
    times = []
    with tf.device(self.device):
      for img in self.x_test[0:self.evaluation_batch_size]:
        start_time = time.time()
        interpreter.set_tensor(input_index, np.expand_dims(img, axis=0).astype(np.float32))
        interpreter.invoke()
        times.append(time.time() - start_time)
        predictions.append(np.argmax(interpreter.tensor(output_index)()[0]))
    predictions = np.array(predictions)
    self.clustered_results['accuracy'] = sum(predictions == self.y_test[0:self.evaluation_batch_size]) \
                                          / len(predictions) * 100
    self.clustered_results['latency'] = sum(times) * 1000 / len(times)
    self.clustered_results['size'] = os.path.getsize(os.path.join(os.path.abspath(self.output_path),
      'clustered_model.tflite')) / 1e6

  def _print_results(self):
    print('---------------------------------')
    print('Baseline model results:')
    print('Accuracy: {:.2f}%'.format(self.baseline_results['accuracy']))
    print('Latency: {:.2f} ms'.format(self.baseline_results['latency']))
    print('Size: {:.2f} MB'.format(self.baseline_results['size']))
    print('---------------------------------')
    print('Quantized model results:')
    print('Accuracy: {:.2f}%'.format(self.quantized_results['accuracy']))
    print('Latency: {:.2f} ms'.format(self.quantized_results['latency']))
    print('Size: {:.2f} MB'.format(self.quantized_results['size']))
    print('---------------------------------')
    print('Pruned model results:')
    print('Accuracy: {:.2f}%'.format(self.pruned_results['accuracy']))
    print('Latency: {:.2f} ms'.format(self.pruned_results['latency']))
    print('Size: {:.2f} MB'.format(self.pruned_results['size']))
    print('---------------------------------')
    print('Clustered model results:')
    print('Accuracy: {:.2f}%'.format(self.clustered_results['accuracy']))
    print('Latency: {:.2f} ms'.format(self.clustered_results['latency']))
    print('Size: {:.2f} MB'.format(self.clustered_results['size']))
    print('---------------------------------')

  def evaluate(self):
    self._evaluate_baseline()
    self._evaluate_quantized()
    self._evaluate_pruned()
    self._evaluate_clustered()
    self._print_results()