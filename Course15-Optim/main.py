import numpy
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10('./dataset',train=True,transform=torchvision.transforms.ToTensor(),download=False)

dataloader = DataLoader(dataset,batch_size=64,shuffle=True,drop_last=False)

class CIFAR10_Diss_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,padding=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1024,64)
        self.linear2 = nn.Linear(64,10)
        self.model2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )
    def forward(self,x):
        x = self.model2(x)
        return x

model = CIFAR10_Diss_model()

loss = nn.CrossEntropyLoss()
optm =torch.optim.SGD(model.parameters(),lr=0.05)
idx = 0

x =[]
y = []
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:

        imgs ,label = data
        output = model(imgs)
        cross_loss = loss(output,label)
        optm.zero_grad()
        cross_loss.backward()
        optm.step()
        running_loss = running_loss+cross_loss
        # idx = idx+1
        # if idx % 100 ==0:
        #     # print(cross_loss)
        #     x.append(idx)
        #     y.append(cross_loss.detach().numpy())
        # print("OK")
    x.append(epoch)
    y.append(running_loss.detach().numpy())

plt.plot(numpy.array(x), numpy.array(y), linestyle='--', color='red', marker='o')
plt.grid(True)
plt.show()