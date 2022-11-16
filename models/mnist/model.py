import torch
import torch.nn as nn

# Define the model
class CNN22(nn.Module):
  def __init__(self):
    super(CNN22, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 5)
    self.conv2 = nn.Conv2d(32, 64, 5)
    self.fc1 = nn.Linear(4 * 4 * 64, 1024)
    self.fc2 = nn.Linear(1024, 10)
  
  # Define the forward pass
  def forward(self, x):
    x = torch.relu(self.conv1(x))
    x = torch.max_pool2d(x, 2, 2)
    x = torch.relu(self.conv2(x))
    x = torch.max_pool2d(x, 2, 2)
    x = x.view(-1, 4 * 4 * 64)
    x = torch.relu(self.fc1(x))
    x = self.fc2(x)
    return x

if __name__ == "__main__":
  # Print a summary of the model
  from torchsummary import summary
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  net = CNN22().to(device)
  summary(net, (1, 28, 28))
