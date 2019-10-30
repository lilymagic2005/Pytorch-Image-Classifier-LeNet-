from __future__ import print_function
import os
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image_normalize_mean = [0.485, 0.456, 0.406]
image_normalize_std = [0.229, 0.224, 0.225]

label_dict = {"not_santa": 0, "santa":1}
train_class_num = len(label_dict)

feature_extract = False

project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
model_save_path = os.path.join(project_path, "saved/model")
