import torch
import torchvision
import torch.nn as nn

dataset = torchvision.datasets.CIFAR10()

input = torch.tensor([[1,-0.5],
                      [-1,3]])

input = torch.reshape(input,(-1,1,2,2))

# print(input.shape)
print(input)

class Activate(nn.Module):
    def __init__(self):
        super().__init__()
        self.activite1 = nn.ReLU()
    def forward(self,x):
        x = self.activite1(x)
        return x

act = Activate()

output = act(input)
print(output)
# print(output.shape)