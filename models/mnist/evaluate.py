import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/mnist')))
from mnist import MNIST
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/mnist')))
from model import CNN22
import matplotlib.pyplot as plt

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
mnist = MNIST()

# Define loaders
test_loader = mnist.get_test_data_loader(batch_size=100, shuffle=True)

# Define the model
net = CNN22()
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
      ax[j].imshow(images[j].cpu().numpy().squeeze(), cmap='gray')
      ax[j].set_title('P: {}, T: {}, C: {:.2f}%'.format(predicted[j].item(), labels[j].item(),
        100 * torch.softmax(outputs, dim=1)[j][predicted[j]].item()))
      ax[j].axis('off')
    plt.show()
    break
