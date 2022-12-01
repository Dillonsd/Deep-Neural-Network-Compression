from ModelEvaluator import ModelEvaluator
from ModelTrainer import ModelTrainer
from itertools import chain
import tensorflow as tf
from PIL import Image
import pandas as pd
import numpy as np
import os

# Define tools from SKLearn to evaluate the model
# SKLearn does not seem to work on the Jetson Nano 2GB
# Reference for code:
# https://stackoverflow.com/questions/47202182/train-test-split-without-using-scikit-learn
def _indexing(x, indices):
  """
  :param x: array from which indices has to be fetched
  :param indices: indices to be fetched
  :return: sub-array from given array and indices
  """
  # np array indexing
  if hasattr(x, 'shape'):
    return x[indices]

  # list indexing
  return [x[idx] for idx in indices]

def train_test_split(*arrays, test_size=0.25, shufffle=True, random_seed=1):
  """
  splits array into train and test data.
  :param arrays: arrays to split in train and test
  :param test_size: size of test set in range (0,1)
  :param shufffle: whether to shuffle arrays or not
  :param random_seed: random seed value
  :return: return 2*len(arrays) divided into train ans test
  """
  # checks
  assert 0 < test_size < 1
  assert len(arrays) > 0
  length = len(arrays[0])
  for i in arrays:
    assert len(i) == length

  n_test = int(np.ceil(length*test_size))
  n_train = length - n_test

  if shufffle:
    perm = np.random.RandomState(random_seed).permutation(length)
    test_indices = perm[:n_test]
    train_indices = perm[n_test:]
  else:
    train_indices = np.arange(n_train)
    test_indices = np.arange(n_train, length)

  return list(chain.from_iterable((_indexing(x, train_indices), _indexing(x, test_indices)) for x in arrays))

def CNN42():
  # Define the model architecture
  # For GTSRB we use a CNN with four convolutional layers and two fc layers
  return tf.keras.models.Sequential([    
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(30,30,3)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.Dense(43, activation='softmax')
  ])

class GTSRB:
  def run(self):
    # Check if we're running on the Jetson Nano 2GB
    jetson = False
    if os.path.exists('/etc/nv_tegra_release'):
      with open('/etc/nv_tegra_release', 'r') as f:
        if 'aarch64' in f.read():
          jetson = True

    # Load the GTSRB dataset
    all_images = []
    all_labels = []

    data_path = os.path.join(os.path.dirname(__file__), 'gtsrb_data')

    # Loop through Train data folders if we're not on the Jetson Nano 2GB
    if not jetson:
      for i in range(len(os.listdir(os.path.join(data_path, 'Train')))):
        path = os.path.join(os.path.join(data_path, 'Train'), str(i))
        images = os.listdir(path)
        
        for image in images:
          try:
            img = Image.open(os.path.join(path, image))
            img = img.resize((30,30))
            all_images.append(np.array(img))
            all_labels.append(i)
          except:
            print("Failed to load " + os.path.join(path, image))

      train_images, train_labels = np.array(all_images), np.array(all_labels)

      # Shuffle the data
      indexes = np.arange(train_images.shape[0])
      np.random.shuffle(indexes)
      images = train_images[indexes]
      labels = train_labels[indexes]

      # Split the data into train and validation sets
      x_train, x_val, y_train, y_val = train_test_split(images, labels, shuffle=True)
      x_train, x_val = x_train / 255.0, x_val / 255.0

      y_train = tf.keras.utils.to_categorical(y_train, len(os.listdir(os.path.join(data_path, 'Train'))))
      y_val = tf.keras.utils.to_categorical(y_val, len(os.listdir(os.path.join(data_path, 'Train'))))

    # Load the test data
    test = pd.read_csv(os.path.join(data_path, 'Test.csv'))
    y_test = test["ClassId"].values
    test_image_paths = test["Path"].values

    test_images = []

    for image in test_image_paths:
      try:
        img = Image.open(os.path.join(data_path, image))
        img = img.resize((30, 30))
        test_images.append(np.array(img))
      except:
        print("Failed to load " + os.path.join(data_path, image))

    x_test = np.array(test_images)
    x_test = x_test / 255

    # Attempt training the model if we're not on the Jetson Nano 2GB
    if not jetson:
      model_paths = [ os.path.join(os.path.dirname(__file__), 'gtsrb/baseline_model.h5'),
                      os.path.join(os.path.dirname(__file__), 'gtsrb/quantized_model.tflite'),
                      os.path.join(os.path.dirname(__file__), 'gtsrb/pruned_model.tflite'),
                      os.path.join(os.path.dirname(__file__), 'gtsrb/clustered_model.tflite') ]
      
      # Check if any of the models already exist
      if not any([os.path.exists(path) for path in model_paths]):
        # Train the models
        trainer = ModelTrainer(CNN42, x_train, y_train, x_test, y_test, 'adam',
          'sparse_categorical_crossentropy', ['accuracy'],
          os.path.join(os.path.dirname(__file__), 'gtsrb'),
          x_val, y_val)
        trainer.train()
    
    # Evaluate the models
    evaluator = ModelEvaluator(CNN42, x_test, y_test,
      os.path.join(os.path.dirname(__file__), 'gtsrb'), 'cpu')
    evaluator.evaluate()

if __name__ == '__main__':
  gtsrb = GTSRB()
  gtsrb.run() 
