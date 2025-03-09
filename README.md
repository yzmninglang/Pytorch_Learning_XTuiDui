# PyTorch 学习项目

## 简介
本项目旨在通过实践学习 PyTorch 框架，涵盖了从基础到高级概念的学习与应用。通过一系列教程和练习，你将能够掌握如何构建、训练和优化深度学习模型。

## 目录结构

以下是本项目的主要目录结构及其用途说明，帮助你快速了解和导航到所需的文件。



### Course08-Conv2d

目录结构：[Course08-Conv2d](./Course08-Conv2d)，此次实验中主要涉及到reshape的用法，其具体内容需要参考[torch.reshape — PyTorch 2.6 documentation](https://pytorch.org/docs/stable/generated/torch.reshape.html)以及add_images的用法



由于Course08中设置的out_channels大于3，对于SummaryWriter无法显示，因此需要用到reshape修改tensor的大小，此外，对于SummaryWriter的来说，由于add_image针对的**一个图片**，接受CHW图片，但是对于batch来说，格式为**NCHW**，所以需要使用**add_images**



### Course09-Pool

[Course09-Pool](./Course09-Pool),对于池化来说，一定要注意nn.MaxPool2d只接受tensor元素类型为float的类型，因此需要提前将其转化为float32，方法是在tensor(...,dtype=torch.float32),除此之外，注意ceilmode的设置



###  [Course10-Activate](Course10-Activate) 

 [Course10-Activate](Course10-Activate) 主要是介绍了激活函数的用法，本身没有十分特别的，注意常规操作即可



###  [Course11-Linear](Course11-Linear) 

这一节主要是涉及到线性层的特点，线性层主要要注意nn.Linear需要添加两个参数，输入的维度和输出维度，除此之外，需要注意Flatten的展开方式，可以参考

- [torch.flatten — PyTorch 2.6 documentation](https://pytorch.org/docs/stable/generated/torch.flatten.html)
- [Pytorch 中 torch.flatten() 和 torch.nn.Flatten() 实例详解 - 别关注我了，私信我吧 - 博客园](https://www.cnblogs.com/BlairGrowing/p/16074632.html)

