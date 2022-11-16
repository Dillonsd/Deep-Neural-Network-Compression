import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/cifar10')))
from cifar10 import CIFAR10
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/cifar10')))
from model import ResNet18
import matplotlib.pyplot as plt

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
cifar10 = CIFAR10()

# Define loaders
test_loader = cifar10.get_test_data_loader(batch_size=128, shuffle=True)

# Define the model
net = ResNet18()
# Load the model
net.load_state_dict(torch.load(
  os.path.abspath(os.path.join(os.path.dirname(__file__), 'model.pth'))))
net.to(device)

# Test the network and show some results
correct = 0
total = 0
with torch.no_grad():
  for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
print('Accuracy: {}%'.format(100 * correct / total))

# Test network prediction latency
import time
start = time.time()
for images, labels in test_loader:
  images = images.to(device)
  labels = labels.to(device)
  outputs = net(images)
end = time.time()
print('Latency: {}ms'.format((end - start) * 1000 / len(test_loader)))

# Verify model size
params = 0
buffers = 0
for p in net.parameters():
  params += p.numel() * p.element_size()
for b in net.buffers():
  buffers += b.numel() * b.element_size()
print('Model size: {}MB'.format((params + buffers) / (1024**2)))

# Display some images with their predicted labels, true labels and confidence
fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(5, 1))
with torch.no_grad():
  for i, (images, labels) in enumerate(test_loader):
    images = images.to(device)
    labels = labels.to(device)
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    for j in range(5):
      ax[j].imshow(images[j].cpu().permute(1, 2, 0))
      ax[j].set_title('P: {}, T: {}, C: {:.2f}%'.format(predicted[j].item(), labels[j].item(),
        100 * torch.softmax(outputs, dim=1)[j][predicted[j]].item()))
      ax[j].axis('off')
    plt.show()
    break
plt.show()
