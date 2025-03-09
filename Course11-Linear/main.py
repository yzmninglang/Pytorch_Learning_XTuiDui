import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn


dataset = torchvision.datasets.CIFAR10('./dataset',train=False,transform=transforms.ToTensor(),download=False)
dataloader = DataLoader(dataset,batch_size=64,shuffle=True,drop_last=True)


for data in dataloader:
    imgs ,label = data
    print(imgs.shape)
    output = torch.reshape(imgs,(1,1,1,1,-1))
    print(output.shape)
    
    
class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(196608,10)

    def forward(self,x):
        x = self.linear(x)
        return x

model = Linear()

for data in dataloader:
    imgs ,label = data
    print(imgs.shape)
    # output = torch.reshape(imgs,(1,1,1,1,-1))
    output = torch.flatten(imgs)
    output =model(output)
    print(output.shape)