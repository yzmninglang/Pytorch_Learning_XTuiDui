import torch.optim
import  torchvision
from torch.utils.data import DataLoader
import  torch.nn as nn

train_dataset = torchvision.datasets.CIFAR10('./dataset',train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_dataset = torchvision.datasets.CIFAR10('./dataset',train=False,transform=torchvision.transforms.ToTensor(),download=True)


print("训练数据集的长度为：",len(train_dataset))
print("测试数据集的长度为：",len(test_dataset))


train_data = DataLoader(train_dataset,batch_size=64,shuffle=True)
test_data = DataLoader(test_dataset,batch_size=64,shuffle=True)



class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 =  nn.Sequential(
            nn.Conv2d(3,32,kernel_size=5,padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32,out_channels=64,padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(1024,64),
            nn.Linear(64,10)
        )

    def forward(self,x):
        return self.model1(x)




#
model1 = model()

cross_loss =nn.CrossEntropyLoss()
optm = torch.optim.SGD(model1.parameters(),lr=0.01)


x = []
y = []
total_step =0

for epoch in range(50):
    print("-----------第{}轮开始-----------".format(epoch))
    x.append(x)
    run_loss = 0.0
    for data in train_data:

        imgs ,label = data
        output = model1(imgs)
        loss = cross_loss(output,label)

        optm.zero_grad()
        cross_loss.backward()
        optm.step()
        # 总数据加1
        total_step = total_step+1
        print("训练次数：{}，Loss：{}".format(total_step,loss.item()))
    # y.append(run_loss)
    total_test_loss = 0
    with torch.no_grad():
        for data in test_data:
            imgs, label =data
            output = model1(data)
            loss =cross_loss(output,label)
            total_test_loss = total_test_loss+loss
    print("测试集上的Loss:{}".format(total_test_loss))
torch.save(model1.state_dict(),"model1.pth")
