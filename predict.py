'''
 - Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.
 - Basic usage: python predict.py /path/to/image checkpoint
 - Options:
   - Return top KK most likely classes: python predict.py input checkpoint --top_k 3
   - Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
   - Use GPU for inference: python predict.py input checkpoint --gpu
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

# python predict.py /path/to/image checkpoint
parser.add_argument('image_path', default='flowers/test/10/image_07104.jpg', action="store")
parser.add_argument('checkpoint', default='./checkpoint.pth', action="store")

# Return top KK most likely classes: python predict.py input checkpoint --top_k 3
parser.add_argument('--top_k', default=5, dest="top_k", action="store")

# Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')

# Use GPU for inference: python predict.py input checkpoint --gpu
parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

parser = parser.parse_args()

image_path = parser.image_path
checkpoint = parser.checkpoint
top_k = parser.top_k
category_names = parser.category_names
gpu = parser.gpu

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)


# Load the checkpoint
filepath = checkpoint
checkpoint = torch.load(filepath)
model = checkpoint["model"]
#model.load_state_dict(checkpoint['state_dict'])


probs, classes = helper.predict(image_path, model, top_k, gpu)
labels = [cat_to_name[i] for i in classes] 

true_class = image_path.split('/')[2]
print('Predicted Class: ' + cat_to_name[classes[0]])
if(true_class == classes[0]):
    print('Predicted Succesfully.')
else:
    print(f'Wrong Prediction, the Actual Class is : ' + cat_to_name[true_class])
 
print('Class Probability')
for i in range(top_k):
    print(f"Class: {labels[i]} with probability: {probs[i]}.")
    

print('Done.')

