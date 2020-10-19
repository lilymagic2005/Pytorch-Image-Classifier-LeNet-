from __future__ import print_function
import os
import torch

par_dir = p = os.path.abspath('.')
data_dir = p + "/data/cifar10png"

num_of_epoch = 30
batch_size = 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image_normalize_mean = [0.485, 0.456, 0.406]
image_normalize_std = [0.229, 0.224, 0.225]

model_input_size = 32
label_dict = {"airplane": 0, "automobile":1,"bird":2,"cat":3,"deer":4,"dog":5,"frog":6,"horse":7,"ship":8,"truck":9}
train_class_num = len(label_dict)

feature_extract = False

project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
model_save_path = os.path.join(project_path, "saved/model")
