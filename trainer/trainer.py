#import dataset.dataset as dataset

#data_loader = dataset.DataLoader()

#for inp, lab in data_loader.dataloaders_dict["train"]:
#    print(inp)
#    print(lab)
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import numpy as np
import time
import os
import copy
from config import config
import codecs
from dataset.data_preprocess import ImageDataProcess
device = config.device

class ModelTrainer(object):
    def __init__(self, model_type,model,size):
        pass

    def train_model(model, dataloaders , criterion, optimizer, num_epochs=25, is_inception=False, model_save_path="./",log_save_path=""):
        since = time.time()
        val_acc_history = []
        test_acc_history = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 20)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                print(phase)
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                batch_num = 0
                batch_start_time = time.time()
                for inputs, labels in dataloaders[phase]:

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        # if is_inception and phase == 'train':
                        #     outputs, aux_outputs = model(inputs)
                        #     loss1 = criterion(outputs, labels)
                        #     loss2 = criterion(aux_outputs, labels)
                        #     loss = loss1 + 0.4 * loss2
                        # else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    if batch_num % 100 == 0:
                        print("batch num:", batch_num, "cost time:", time.time() - batch_start_time)
                        batch_start_time = time.time()
                    batch_num += 1

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, model_save_path)

                if phase == 'val':
                    val_acc_history.append(epoch_acc)

            epoch_time_cost = time.time() - epoch_start_time
            print("epoch complete in {:.0f}m {:.0f}s".format(epoch_time_cost // 60, epoch_time_cost % 60))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:.4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, val_acc_history