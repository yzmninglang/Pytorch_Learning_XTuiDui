import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn

# dataset = torchvision.datasets.ImageNet('./dataset',split='train',transform=torchvision.transforms.ToTensor(),download=True)
# dataloader = DataLoader(dataset,batch_size=64,shuffle=True,drop_last=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)
vgg16_false = torchvision.models.vgg16(pretrained=False)
print(vgg16_true)

vgg16_true.classifier.add_module('add_linear',nn.Linear(1000,10))
# print(vgg16_false)
vgg16_false.classifier[6] =  nn.Linear(4096,100)
print(vgg16_false)