import torch
import torch.nn as nn


inputs = torch.tensor([1,2,3],dtype=torch.float32)
targets = torch.tensor([1,2,5],dtype=torch.float32)

inputs = torch.reshape(inputs, (1,1,1,3))
targets = torch.reshape(targets,(1,1,1,3))

loss_sum = nn.L1Loss(reduction='sum')
loss_avg = nn.L1Loss(reduction='mean')

loss_mse =nn.MSELoss(reduction='mean')

result_sum_loss = loss_sum(inputs,targets)
result_avg_loss = loss_avg(inputs,targets)
result_avg_mse =loss_mse(inputs,targets)
print(result_sum_loss,result_avg_loss,result_avg_mse)


x = torch.tensor([0.1,0.2,0.3])
y = torch.tensor([1])


x = torch.reshape(x,[1,3,])
loss_cross = nn.CrossEntropyLoss()
result_Loss_cross = loss_cross(x,y)
print(result_Loss_cross)