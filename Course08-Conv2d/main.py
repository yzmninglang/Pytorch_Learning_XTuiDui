



import torchvision
from torch.utils.data import  DataLoader
from torch import  nn


root_dir = './dataset'
dataset_transfoms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

dataset = torchvision.datasets.CIFAR10(root_dir,train=False, transform=dataset_transfoms, download=True)

data_loader = DataLoader(dataset=dataset,batch_size=64,shuffle=True,num_workers=0,drop_last=False)


class MyModule(nn.Module):
    def __int__(self):
        super(self,MyModule).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)

    def forward(self,x):
        self.conv1(x)
        return x

moduler = MyModule()
print(moduler)

for  data in data_loader:
    # 这里是四张图片一起放
    imgs, target = data
    output = moduler(imgs)
    print(imgs.shape)
    print(output.shape)

# print(dataset)
