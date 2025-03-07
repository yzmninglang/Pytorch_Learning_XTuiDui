from torchvision import transforms
# import Image
from PIL import Image
import numpy as np
from torch.utils.tensorboard import SummaryWriter

img_path = "hymenoptera_data/train/ants/0013035.jpg"
img = Image.open(img_path)
# print(img)

writer = SummaryWriter("logs")



# tensor类型
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)



# numpy array类型
# np_img = np.array(img)

writer.add_image("numpy_image", tensor_img, 1)
writer.close()