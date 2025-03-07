
from torch import nn
import torch
class model(nn.Module):
    def __int__(self):
        super().__init__()

    def forward(self,input):
        output = input +1
        return  output

model = model()
x = torch.tensor(8)
out=model(x)
print(out)
# 第一个NN