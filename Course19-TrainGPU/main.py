import numpy as np
import torch.optim
import  torchvision
from torch.utils.data import DataLoader
import  torch.nn as nn
from tqdm import *
import matplotlib.pyplot as plt
from  torch.utils.tensorboard import SummaryWriter


train_dataset = torchvision.datasets.CIFAR10('./dataset',train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_dataset = torchvision.datasets.CIFAR10('./dataset',train=False,transform=torchvision.transforms.ToTensor(),download=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("训练数据集的长度为：",len(train_dataset))
print("测试数据集的长度为：",len(test_dataset))
train_data = DataLoader(train_dataset,batch_size=64,shuffle=True)
test_data = DataLoader(test_dataset,batch_size=64,shuffle=True)
test_loss=[]
train_loss = []

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 =  nn.Sequential(
            nn.Conv2d(3,32,kernel_size=5,padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(1024,64),
            nn.Linear(64,10)
        )

    def forward(self,x):
        return self.model1(x)
#
# model1 = model()
# 将模型转移到cuda上
model_state = torch.load('model/model1_epoch_49.pth')
# print(model1)
model1 = model()
model1.load_state_dict(model_state)
model1.to(device)


cross_loss =nn.CrossEntropyLoss()
# 损失函数转移到cuda上
cross_loss=cross_loss.to(device=device)
optm = torch.optim.SGD(model1.parameters(),lr=0.001)

writer = SummaryWriter("logs")
x = []
# y = []
total_step =0
epoch_num = 50
for epoch in range(epoch_num):
    # print("-----------第{}轮开始-----------".format(epoch))
    x.append(epoch)
    run_loss = 0.0
    loop = tqdm((train_data), total=len(train_data))
    temp_train_loss = 0
    temp_test_loss = 0
    total_step = total_step + 1
    # total_train_step = 0
    accuary_train=[]
    accuary_test = []
    model1.train()
    for data in loop:

        imgs ,label = data
        # 将数据转移到cuda上
        imgs =imgs.to(device)
        label = label.to(device)

        output = model1(imgs)
        loss = cross_loss(output,label)

        optm.zero_grad()
        loss.backward()
        optm.step()
        # 总数据加1
        # print("训练次数：{}，Loss：{}".format(total_step,loss.item()))
        loop.set_description(f'Epoch [{epoch}/{epoch_num}]')
        temp_train_loss = temp_train_loss+loss.item()
        accuary_temp_train = (output.argmax(1) == label).sum() / len(label)
        accuary_train.append(accuary_temp_train)
        loop.set_postfix(loss=loss.item(),acc= torch.tensor(accuary_train).mean().item())


    writer.add_scalar("tarin_loss",temp_train_loss,epoch)
    writer.add_scalar("tarin_accuary", torch.tensor(accuary_train).mean(), epoch)
    train_loss.append(temp_train_loss)

    # y.append(run_loss)


    model1.eval()
    with torch.no_grad():
        for data in test_data:
            imgs, label =data
            imgs =imgs.to(device)
            label = label.to(device)
            output = model1(imgs)
            loss =cross_loss(output,label)
            temp_test_loss = temp_test_loss+loss.item()
            accuary_temp_test = (output.argmax(1)==label).sum()/len(label)
            accuary_test.append(accuary_temp_test)
        test_loss.append(temp_test_loss)
        writer.add_scalar("test_loss",temp_test_loss,epoch)
        writer.add_scalar("test_accuary",torch.tensor(accuary_test).mean(),epoch)


    # 每一轮保存一次模型
    torch.save(model1.state_dict(), "./model/model1_epoch_{}.pth".format(epoch))
    # 每一轮画图
    plt.clf()
    plt.plot(np.array(x), np.array(train_loss))
    plt.plot(np.array(x), np.array(test_loss))
    plt.grid(True)
    plt.show()
    # print("测试集上的Loss:{}".format(total_test_loss))

writer.close()
np.save('train_loss.npy',np.array(train_loss))
np.save('test_loss.npy',np.array(test_loss))


