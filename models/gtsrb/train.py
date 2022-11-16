import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/gtsrb')))
from gtsrb import GTSRB
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/gtsrb')))
from model import ResNet18

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device: {}'.format(device))

# Load the dataset
gtsrb = GTSRB()

# Define loaders
train_loader = gtsrb.get_train_data_loader(batch_size=256, shuffle=True)
test_loader = gtsrb.get_test_data_loader(batch_size=256, shuffle=False)

# Define the model
net = ResNet18().to(device)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

# Define the learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Train method
def train(epoch):
  net.train()
  train_loss = 0
  correct = 0
  total = 0
  for batch_idx, (inputs, targets) in enumerate(train_loader):
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    train_loss += loss.item()
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()
    print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}/{} ({:.2f}%)'.format(
      epoch, batch_idx * len(inputs), len(train_loader.dataset),
      100. * batch_idx / len(train_loader), loss.item(),
      correct, total, 100. * correct / total))

# Test method
def test(epoch):
  net.eval()
  test_loss = 0
  correct = 0
  total = 0
  with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(train_loader):
      inputs, targets = inputs.to(device), targets.to(device)
      outputs = net(inputs)
      loss = criterion(outputs, targets)
      test_loss += loss.item()
      _, predicted = outputs.max(1)
      total += targets.size(0)
      correct += predicted.eq(targets).sum().item()
  print('Epoch: {} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
    epoch, test_loss / (batch_idx + 1), correct, total,
    100. * correct / total))
  return test_loss / (batch_idx + 1)

# Train the network
for epoch in range(1, 100):
  train(epoch)
  test_loss = test(epoch)
  scheduler.step(test_loss)
  if epoch % 50 == 0:
    torch.save(net.state_dict(), f'model_{epoch}.pth')

torch.save(net.state_dict(), 'model.pth')
