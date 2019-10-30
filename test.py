from model import model
import torch
import torch.nn as nn
from config import config
from dataset.data_preprocess import ImageDataProcess
device = config.device
model_path = "saved/model/model.pth"

#--------Load the Model and run as GPU Mode---------------------#
model = model.LeNet(config.train_class_num).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

#-------------------Process Photo--------------------------------#
image_tensor_list = []
image_dir = "data/test/santapepe.jpg"
image_tensor = ImageDataProcess.image_normalize(image_dir) #image to Tensor
image_tensor_list.append(torch.unsqueeze(image_tensor, 0))
input_tensor = torch.cat(tuple(image_tensor_list), dim=0) # +1 Dimension
input_tensor = input_tensor.cuda()

#------------------Softmax the result to probability--------------#
outputs_tensor = model(input_tensor)
m_softmax = nn.Softmax(dim=1)
outputs_tensor = m_softmax(outputs_tensor)

#------------------Print the Result--------------#
print(outputs_tensor)

