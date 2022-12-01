import os
import tensorflow as tf
import tensorflow_model_optimization as tfmot

class ModelTrainer:
  def __init__(self, setup_model, x_train, y_train, x_test, y_test, optimizer, loss,
      metrics, output_path, x_val=None, y_val=None, epochs=50, comp_epochs=1, batch_size=32):
    self.setup_model_func = setup_model
    self.x_train = x_train
    self.y_train = y_train
    self.x_test = x_test
    self.y_test = y_test
    self.optimizer = optimizer
    self.loss = loss
    self.metrics = metrics
    self.output_path = output_path
    self.x_val = x_val
    self.y_val = y_val
    self.epochs = epochs
    self.comp_epochs = comp_epochs
    self.batch_size = batch_size
  
  def _train_baseline(self):
    model = self.setup_model_func()
    model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)]
    if self.x_val is not None:
      model.fit(self.x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size,
        validation_data=(self.x_val, self.y_val), callbacks=callbacks)
    else:
      model.fit(self.x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size,
        validation_split=0.2, callbacks=callbacks)
    model.save(os.path.join(os.path.abspath(self.output_path), 'baseline_model.h5'))

  def _train_quantized(self):
    model = self.setup_model_func()
    model.load_weights(os.path.join(os.path.abspath(self.output_path), 'baseline_model.h5'))
    q_aware_model = tfmot.quantization.keras.quantize_model(model)
    q_aware_model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
    if self.x_val is not None:
      q_aware_model.fit(self.x_train, self.y_train, epochs=self.comp_epochs, batch_size=self.batch_size,
        validation_data=(self.x_val, self.y_val))
    else:
      q_aware_model.fit(self.x_train, self.y_train, epochs=self.comp_epochs, batch_size=self.batch_size,
        validation_split=0.2)
    converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    q_model = converter.convert()

    with open(os.path.join(os.path.abspath(self.output_path), 'quantized_model.tflite'), 'wb') as f:
      f.write(q_model)
  
  def _train_pruned(self):
    model = self.setup_model_func()
    model.load_weights(os.path.join(os.path.abspath(self.output_path), 'baseline_model.h5'))
    pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
        final_sparsity=0.80, begin_step=0, end_step=1000)
    }
    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
    model_for_pruning.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
    if self.x_val is not None:
      model_for_pruning.fit(self.x_train, self.y_train, epochs=self.comp_epochs, batch_size=self.batch_size,
        validation_data=(self.x_val, self.y_val), callbacks=callbacks)
    else:
      model_for_pruning.fit(self.x_train, self.y_train, epochs=self.comp_epochs, batch_size=self.batch_size,
        validation_split=0.2, callbacks=callbacks)
    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    p_model = converter.convert()

    with open(os.path.join(os.path.abspath(self.output_path), 'pruned_model.tflite'), 'wb') as f:
      f.write(p_model)
  
  def _train_clustered(self):
    model = self.setup_model_func()
    model.load_weights(os.path.join(os.path.abspath(self.output_path), 'baseline_model.h5'))
    clustered_model = tfmot.clustering.keras.cluster_weights(model, number_of_clusters=8,
      cluster_centroids_init=tfmot.clustering.keras.CentroidInitialization.KMEANS_PLUS_PLUS)
    clustered_model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
    if self.x_val is not None:
      clustered_model.fit(self.x_train, self.y_train, epochs=self.comp_epochs, batch_size=self.batch_size,
        validation_data=(self.x_val, self.y_val))
    else:
      clustered_model.fit(self.x_train, self.y_train, epochs=self.comp_epochs, batch_size=self.batch_size,
        validation_split=0.2)
    stripped_model = tfmot.clustering.keras.strip_clustering(clustered_model)
    converter = tf.lite.TFLiteConverter.from_keras_model(stripped_model)
    clustered_model = converter.convert()

    with open(os.path.join(os.path.abspath(self.output_path), 'clustered_model.tflite'), 'wb') as f:
      f.write(clustered_model)

  def train(self):
    self._train_baseline()
    self._train_quantized()
    self._train_pruned()
    self._train_clustered()
