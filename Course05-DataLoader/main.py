import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

writer = SummaryWriter("logs")
dataset_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

root_dir="./dataset"
trainset = torchvision.datasets.CIFAR10(root_dir, train=True, transform=dataset_transforms, download=True)
# 准备的测试集
testset = torchvision.datasets.CIFAR10(root_dir, train=False, transform=dataset_transforms, download=True)

test_loader = DataLoader(dataset=testset, batch_size=64, shuffle=False, num_workers=0, drop_last=False)

# 测试集的第一个图片及target
img, target = testset[0]
print(img.shape)
print(target)
# print(type(test_loader))
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, target = data
        writer.add_images("Epoch:{}".format(epoch), imgs, step)
        step = step+1
writer.close()
    # print("---------------------------")
