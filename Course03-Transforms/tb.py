from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
img = Image.open("image/1_LLVL8xUiUOBE8WHgzAuY-Q.png")
trans_tensor = transforms.ToTensor()
# transforms.__call__函数
img_tensor = trans_tensor(img)
writer.add_image("Tensor_image", img_tensor, 1)


# 图片归一化
# print(img_tensor[0][0][0])
trans_Normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_nor = trans_Normalize(img_tensor)
# print(img_nor[0][0][0])
writer.add_image("img_nor", img_nor, 1)



# resize
# print(type(img))
trans_resize = transforms.Resize((500, 500))
img_resize = trans_resize(img_tensor)
writer.add_image("img_resize", img_resize, 1)



# RandCrop :随机截取图像中指定大小的部分，将其作为新的图像
trans_RandCrop = transforms.RandomCrop(1000)
trans_RandCrop_2 = transforms.Compose([trans_RandCrop, trans_tensor])

# print("Size", img_randcrop.shape)
for i in range(10):
    img_randcrop = trans_RandCrop_2(img)
    writer.add_image("Rand_Crop", img_randcrop, i)

# 用于连接两个操作，输入一个列表，结果顺序执行

# Compose  -- resize --2
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_tensor])
img_reszie_2 = trans_compose(img)

# print(img_reszie_2.shape)
writer.add_image("Compose_Resize", img_reszie_2, 1)


writer.close()


