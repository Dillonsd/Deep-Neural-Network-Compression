from ModelEvaluator import ModelEvaluator
from ModelTrainer import ModelTrainer
import tensorflow as tf
import os

def CNN22():
  # Define the model architecture
  # For MNIST we use a simple CNN with two convolutional layers and two fc layers
  return tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(28, 28)),
    tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, (5, 5), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(10)
  ])

class MNIST:
  def run(self):
    # Check if we're running on the Jetson Nano 2GB
    jetson = False
    if os.path.exists('/etc/nv_tegra_release'):
      with open('/etc/nv_tegra_release', 'r') as f:
        if 'aarch64' in f.read():
          jetson = True

    # Load the MNIST dataset
    mnist_dataset = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist_dataset.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Attempt to train the model if we're not on the Jetson Nano 2GB
    if not jetson:
      model_paths = [ os.path.join(os.path.dirname(__file__), 'mnist/baseline_model.h5'),
                      os.path.join(os.path.dirname(__file__), 'mnist/quantized_model.tflite'),
                      os.path.join(os.path.dirname(__file__), 'mnist/pruned_model.tflite'),
                      os.path.join(os.path.dirname(__file__), 'mnist/clustered_model.tflite') ]

      # Check if none of the models exist
      if not any([os.path.exists(path) for path in model_paths]):
        # Train the models
        trainer = ModelTrainer(CNN22, x_train, y_train, x_test, y_test, 'adam',
          'sparse_categorical_crossentropy', ['accuracy'], os.path.join(os.path.dirname(__file__), 'mnist'))
        trainer.train()
    
    # Evaluate the models
    evaluator = ModelEvaluator(CNN22, x_test, y_test,
      os.path.join(os.path.dirname(__file__), 'mnist'), 'cpu')
    evaluator.evaluate()

if __name__ == '__main__':
  mnist = MNIST()
  mnist.run()
  