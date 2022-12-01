from ModelEvaluator import ModelEvaluator
from ModelTrainer import ModelTrainer
import tensorflow as tf
import numpy as np
import os

def CNN42():
  # Define the model architecture
  # For CIFAR10 we use a CNN with four convolutional layers and two fc layers
  return tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, 3, padding='same', input_shape=(32, 32, 3), activation='relu'),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax'),
  ])

class CIFAR10:
  def run(self):
    # Check if we're running on the Jetson Nano 2GB
    jetson = False
    if os.path.exists('/etc/nv_tegra_release'):
      with open('/etc/nv_tegra_release', 'r') as f:
        if 'aarch64' in f.read():
          jetson = True

    # Load the CIFAR10 dataset
    cifar10_dataset = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10_dataset.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_test = np.array([y[0] for y in y_test])

    # Attempt to train the model if we're not on the Jetson Nano 2GB
    if not jetson:
      model_paths = [ os.path.join(os.path.dirname(__file__), 'cifar10/baseline_model.h5'),
                      os.path.join(os.path.dirname(__file__), 'cifar10/quantized_model.tflite'),
                      os.path.join(os.path.dirname(__file__), 'cifar10/pruned_model.tflite'),
                      os.path.join(os.path.dirname(__file__), 'cifar10/clustered_model.tflite') ]
      
      # Check if none of the models exist
      if not any([os.path.exists(path) for path in model_paths]):
        # Train the models
        trainer = ModelTrainer(CNN42, x_train, y_train, x_test, y_test, 'adam',
          'sparse_categorical_crossentropy', ['accuracy'], os.path.join(os.path.dirname(__file__), 'cifar10'))
        trainer.train()
    
    # Evaluate the models
    evaluator = ModelEvaluator(CNN42, x_test, y_test,
      os.path.join(os.path.dirname(__file__), 'cifar10'), 'cpu', 10)
    evaluator.evaluate()

if __name__ == '__main__':
  cifar10 = CIFAR10()
  cifar10.run()