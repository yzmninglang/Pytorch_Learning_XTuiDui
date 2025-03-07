# PyTorch 学习项目

## 简介
本项目旨在通过实践学习 PyTorch 框架，涵盖了从基础到高级概念的学习与应用。通过一系列教程和练习，你将能够掌握如何构建、训练和优化深度学习模型。

## 目录结构

以下是本项目的主要目录结构及其用途说明，帮助你快速了解和导航到所需的文件。



### Course08-Conv2d

目录结构：[Course08-Conv2d](./Course08-Conv2d)，此次实验中主要涉及到reshape的用法，其具体内容需要参考[torch.reshape — PyTorch 2.6 documentation](https://pytorch.org/docs/stable/generated/torch.reshape.html)以及add_images的用法



由于Course08中设置的out_channels大于3，对于SummaryWriter无法显示，因此需要用到reshape修改tensor的大小，此外，对于SummaryWriter的来说，由于add_image针对的**一个图片**，接受CHW图片，但是对于batch来说，格式为**NCHW**，所以需要使用**add_images**



