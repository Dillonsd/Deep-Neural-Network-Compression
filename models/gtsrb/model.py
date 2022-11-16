from torchvision.models import resnet18
import torch

def ResNet18():
  return resnet18()

if __name__ == "__main__":
  # Print a summary of the model
  from torchsummary import summary
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  net = ResNet18().to(device)
  summary(net, (3, 32, 32))