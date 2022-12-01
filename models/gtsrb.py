from sklearn.model_selection import train_test_split
from ModelEvaluator import ModelEvaluator
from ModelTrainer import ModelTrainer
import tensorflow as tf
from PIL import Image
import pandas as pd
import numpy as np
import os

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
  
  def run(self):
    # Load the GTSRB dataset
    all_images = []
    all_labels = []

    data_path = os.path.join(os.path.dirname(__file__), 'gtsrb_data')

    # Loop through Train data folders
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
      os.path.join(os.path.dirname(__file__), 'gtsrb'), 'cpu', 10)
    evaluator.evaluate()

if __name__ == '__main__':
  gtsrb = GTSRB()
  gtsrb.run() 
