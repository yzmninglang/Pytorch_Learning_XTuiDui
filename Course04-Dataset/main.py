import torchvision
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter("logs")
dataset_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

root_dir="./dataset"
trainset = torchvision.datasets.CIFAR10(root_dir, train=True, transform=dataset_transforms, download=True)
testset = torchvision.datasets.CIFAR10(root_dir, train=False, transform=dataset_transforms, download=True)

# print(trainset.classes)
for i in range(10):
    img , target = trainset[i]
    writer.add_image("train", img, i)
writer.close()

for i in range(10):
    img , target = testset[i]
    writer.add_image("test", img, i)
writer.close()

# print((i))
print(target)