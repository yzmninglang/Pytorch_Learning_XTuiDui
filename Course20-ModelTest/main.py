import torch
from PIL import Image
from torchvision.transforms import *
from  Model import *
# import torch

img_path = 'image/003.png'
img = Image.open(img_path)
img = img.convert('RGB')

# 需要和训练时的尺寸一致
transfom = Compose([Resize((32,32)),
                    ToTensor()])
img = transfom(img)
print(img.shape)

model = model()
model_dict = torch.load('model/model1_epoch_49.pth')
model.load_state_dict(model_dict)
img = torch.reshape(img,(1,3,32,32))
# print(model(img))
model.eval()
with torch.no_grad():
    output = model(img)
print(output.argmax(1))
