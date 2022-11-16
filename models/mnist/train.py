import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/mnist')))
from mnist import MNIST
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/mnist')))
from model import CNN22

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
mnist = MNIST()

# Define loaders
train_loader = mnist.get_train_data_loader(batch_size=100, shuffle=True)

# Define the model
net = CNN22().to(device)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# Train the network
for epoch in range(50):
  for i, (images, labels) in enumerate(train_loader):
    images = images.to(device)
    labels = labels.to(device)
    # Forward pass
    outputs = net(images)
    loss = criterion(outputs, labels)
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # Print statistics
    if (i + 1) % 100 == 0:
      print('Epoch: {}/{}, Step: {}/{}, Loss: {}'.format(epoch + 1, 50, i + 1, len(train_loader), loss.item()))
  # Save the model checkpoint every 10 epochs
  if (epoch + 1) % 10 == 0:
    torch.save(net.state_dict(), f'model_{epoch + 1}.pth')

torch.save(net.state_dict(), 'model.pth')
