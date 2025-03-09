import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
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
    #     使用sequential
        self.model1 = nn.Sequential(self.conv1,
                                     self.maxpool1,
                                     self.conv2,
                                     self.maxpool2,
                                     self.conv3,
                                     self.maxpool3,
                                     self.flatten,
                                     self.linear1,
                                     self.linear2)
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
        # 历史写法
        # x= self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # # 由于flatten的默认参数是从start_dim=1开始展开，所以处于第0维度的（图片个数，bathsize对应的那个数）没有被展开
        # x = self.flatten(x)
        # x = self.linear1(x)
        #
        # x = self.linear2(x)


        # 引入sequential
        x = self.model2(x)
        return x

model = CIFAR10_Diss_model()

loss = nn.CrossEntropyLoss()

# imgs, label = dataloader[0]
# print(imgs, label)
for data in dataloader:
    imgs ,label = data
    # print(imgs.shape,label)
    output = model(imgs)
    cross_loss = loss(output,label)
    cross_loss.backward()
    print("OK")

