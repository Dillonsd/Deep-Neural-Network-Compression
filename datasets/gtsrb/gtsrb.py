import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import numpy as np
import torch
import os

# Image transforms
data_transforms = transforms.Compose([
	  transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# Class for loading the CIFAR10 dataset
class GTSRB():
  def __init__(self):
    # Download training data from open datasets.
    self.train_data = datasets.GTSRB(
      root=os.path.abspath(os.path.join(os.path.dirname(__file__), 'train')),
      split='train',
      download=True,
      transform=data_transforms,
    )
    self.test_data = datasets.GTSRB(
      root=os.path.abspath(os.path.join(os.path.dirname(__file__), 'test')),
      split='test',
      download=True,
      transform=data_transforms,
    )
  
  def get_train_data_loader(self, batch_size, shuffle):
    return torch.utils.data.DataLoader(
      self.train_data,
      batch_size=batch_size,
      shuffle=shuffle,
      num_workers=0,
    )
  
  def get_test_data_loader(self, batch_size, shuffle):
    return torch.utils.data.DataLoader(
      self.test_data,
      batch_size=batch_size,
      shuffle=shuffle,
      num_workers=0,
    )
  
  def plot_random_sample(self, n):
    # Plot a random sample of n images
    fig, ax = plt.subplots(1, n, figsize=(20, 2))
    for i in range(n):
      idx = np.random.randint(0, len(self.train_data))
      img, label = self.train_data[idx]
      ax[i].imshow(img.permute(1, 2, 0))
      ax[i].set_title(f'Label: {label}')
    plt.show()

if __name__ == "__main__":
  # Test the class and plot a random sample of 5 images
  gtsrb = GTSRB()
  gtsrb.plot_random_sample(5)
