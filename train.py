from model import model
from trainer import trainer
from dataset import dataset
from config import config
import torch.nn as nn
import torch.optim as optim
import time
import os

device = config.device
feature_extract = config.feature_extract
if __name__ == "__main__":

    model = model.LeNet(config.train_class_num).to(device)

    params_to_update = model.parameters()
    print("Params to learn:")

    if feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print("\t", name)

    optimizer_ft = optim.Adam(params_to_update, lr=0.001)
    criterion = nn.CrossEntropyLoss()
    data_loader = dataset.DataLoader()

    trainer.ModelTrainer.train_model(model,data_loader.dataloaders_dict,criterion,optimizer_ft, num_epochs=config.num_of_epoch ,is_inception=False, model_save_path=os.path.join(config.model_save_path,"model.pth"))