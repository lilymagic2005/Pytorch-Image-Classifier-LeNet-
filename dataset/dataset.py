from __future__ import print_function
import torch
from torch.utils.data import Dataset
from .data_preprocess import ImageDataProcess
import os
import config.config as config

data_dir = "../data"
batch_size = 10

class TheDataset(Dataset):
    def __init__(self, img_path, img_label_dict):

        self.class_folder_dir = [] #Get the folder directory of all the class
        self.class_folder_num = [] #The label of the class , For example : if we've 3 class , then [0,1,2]
        for class_name , class_number in img_label_dict.items():
            self.class_folder_dir.append(os.path.join(img_path,class_name))
            self.class_folder_num.append(class_number)

        self.class_image_dir = [] #Get all the images from all the class
        self.class_image_num = [] #Give a corresponding class

        for class_name , class_number in zip(self.class_folder_dir, self.class_folder_num):
            class_whole_image_dir = os.listdir(os.path.join(class_name))
            for img_dir in class_whole_image_dir:
                self.class_image_dir.append(os.path.join(class_name,img_dir))
                self.class_image_num.append(class_number)

    def __len__(self):
        return len(self.class_image_dir)

    def __getitem__(self , item):
        image_dir = self.class_image_dir[item]
        label = self.class_image_num[item]
        try:
            image_object = ImageDataProcess.image_normalize(image_dir)
            return image_object , label
        except Exception as e:
            print("Error Reading ", image_dir, e)


class DataLoader(object):
    def __init__(self):
        self.image_datasets = {x: TheDataset(os.path.join(data_dir, x ),config.label_dict)
                               for x in ['train','val']}
        self.dataloaders_dict = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=1)
                            for x in ['train', 'val',]}