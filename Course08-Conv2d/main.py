import torch
import torchvision
from torch.utils.data import  DataLoader
from torch import  nn
from torch.utils.tensorboard import SummaryWriter



root_dir = './dataset'
dataset_transfoms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

dataset = torchvision.datasets.CIFAR10(root_dir,train=False, transform=dataset_transfoms, download=True)

data_loader = DataLoader(dataset=dataset,batch_size=64,shuffle=True,num_workers=0,drop_last=False)


class MyModule(nn.Module):
    # 这个地方一定要注意init的写法，不然code无法work
    def __init__(self):
        super(MyModule,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)

    def forward(self,x):
        x = self.conv1(x)
        return x

moduler = MyModule()
print(moduler)
writer = SummaryWriter("logs")
idx =0
for data in data_loader:
    # 这里是四张图片一起放
    imgs, target = data
    output = moduler(imgs)
    # print(output.shape)
    writer.add_images("Original",imgs,idx )

    # torch.size(64,6,30,30) -> [xxx,3,30,30]
    output=torch.reshape(output,(-1,3,30,30))
    writer.add_images("Output", output, idx)

    idx = idx+1

writer.close()
# # print(dataset)
