import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import  DataLoader
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter("logs")
input = torch.tensor([[1,2,0,3,1],
                    [0,1,2,3,1],
                    [1,2,1,0,0],
                    [5,2,3,1,1],
                    [2,1,0,1,1]],dtype=torch.float32)
input = torch.reshape(input,(-1,1,5,5))
print(input)

dataset  = torchvision.datasets.CIFAR10('./dataset',train=False,transform=torchvision.transforms.ToTensor(),download=False)
data_loader = DataLoader(dataset=dataset,batch_size=64,shuffle=True,num_workers=0,drop_last=False)
class Pool_nn(nn.Module):

    def __init__(self):
        super(Pool_nn,self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self,x):
        x = self.maxpool1(x)
        return x

pool =Pool_nn()
# output = pool(input)
# print(output)
idx =0
for data in data_loader:
    imgs , target = data
    output = pool(imgs)
    writer.add_images("input",imgs,idx)
    writer.add_images("output",output,idx)
    idx = idx +1
    # print(output.shape)


