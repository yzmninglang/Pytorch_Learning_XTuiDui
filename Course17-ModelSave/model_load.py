import torch
import torchvision

# 方式1，加载模型
model1 = torch.load("vgg16_method1.pth")
# print()

vgg16 = torchvision.models.vgg16(pretrained=False)

model2 = torch.load("vgg16_method2.pth")
vgg16.load_state_dict(model2)
print(model2)


# model save && model load