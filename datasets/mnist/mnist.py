import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import torch
import os

# Class for loading the MNIST dataset
class MNIST():
  def __init__(self):
    # Download training data from open datasets.
    self.train_data = datasets.MNIST(
      root=os.path.abspath(os.path.join(os.path.dirname(__file__), 'train')),
      train=True,
      download=True,
      transform=ToTensor(),
    )
    self.test_data = datasets.MNIST(
      root=os.path.abspath(os.path.join(os.path.dirname(__file__), 'test')),
      train=False,
      download=True,
      transform=ToTensor(),
    )
  
  def get_train_data_loader(self, batch_size, shuffle):
    return torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, shuffle=shuffle)
  
  def get_test_data_loader(self, batch_size, shuffle):
    return torch.utils.data.DataLoader(self.test_data, batch_size=batch_size, shuffle=shuffle)
  
  def plot_random_sample(self, n):
    # Plot a random sample of n images
    fig, ax = plt.subplots(nrows=1, ncols=n, figsize=(n, 1))
    for i in range(n):
      idx = np.random.randint(0, len(self.train_data))
      img, label = self.train_data[idx]
      ax[i].imshow(img.squeeze(), cmap="gray")
      ax[i].set_title(label)
      ax[i].axis("off")
    plt.show()

if __name__ == "__main__":
  # Test the class and plot a random sample of 5 images
  mnist = MNIST()
  mnist.plot_random_sample(5)
