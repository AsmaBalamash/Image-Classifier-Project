'''
Train a new network on a data set with train.py

Basic usage: 
 - python train.py data_directory
 - Prints out training loss, validation loss, and validation accuracy as the network trains
Options:
 - Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
 - Choose architecture: python train.py data_dir --arch "vgg13"
 - Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
 - Use GPU for training: python train.py data_dir --gpu

'''

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
import json
import helper

#The best way to get the command line input into the scripts is with the argparse module 
import argparse 

parser = argparse.ArgumentParser()

# python train.py data_directory
parser.add_argument('data_directory', action='store', default = './flowers/')

# Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
parser.add_argument('--save_dir', dest="save_dir", action="store", default=".")

# Choose architecture: python train.py data_dir --arch "vgg13" , default is "densenet121"
parser.add_argument('--arch', dest="arch", action="store", default="densenet121")

# Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.003)
parser.add_argument('--hidden_units', dest="hidden_units", action="store", default=512)
parser.add_argument('--epochs', dest="epochs", action="store", default=5)

# Use GPU for training: python train.py data_dir --gpu
parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")


parser = parser.parse_args()
data_dir = parser.data_directory
save_dir = parser.save_dir
arch = parser.arch
learning_rate = parser.learning_rate
hidden_units = parser.hidden_units
epochs = parser.epochs
gpu = parser.gpu

# 1) Load Data
image_datasets, trainloader, testloader, validloader = helper.loadData(data_dir)
# 2) Build Model
model = helper.build_model(arch, hidden_units)
# 3) Train Model
model, optimizer, criterion = helper.train_model(model, trainloader, validloader, learning_rate, epochs, gpu)
# 4) Save the checkpoint 
model.to('cpu')
model.class_to_idx = image_datasets['train_data'].class_to_idx
checkpoint = {'model': model,
              'hidden_units': hidden_units,
              'optimizer_state_dict': optimizer.state_dict,
              'criterion': criterion,
              'epochs': epochs,
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx}

torch.save(checkpoint, save_dir + '/checkpoint.pth')


if save_dir == ".":
    dir_name = "current folder"
else:
    dir_name = save_dir + " folder"

print(f'File saved to {dir_name} with name (checkpoint.pth).')
