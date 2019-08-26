# Imports here
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
from collections import OrderedDict
from workspace_utils import active_session



# Load the data
def loadData(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets
    # data_transforms
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]) 
    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    image_datasets = {}
    image_datasets["train_data"] = datasets.ImageFolder(train_dir, transform=train_transforms)
    image_datasets["test_data"] = datasets.ImageFolder(test_dir, transform=test_transforms)
    image_datasets["valid_data"] = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    # dataloaders = 
    trainloader = torch.utils.data.DataLoader(image_datasets["train_data"], batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(image_datasets["test_data"], batch_size=64)
    validloader = torch.utils.data.DataLoader(image_datasets["valid_data"], batch_size=64)
    print(f"Data Loaded Successfully from {data_dir}.")
    return image_datasets, trainloader, testloader, validloader


def build_model(arch, hidden_units):
     #1) Load a pre-trained network 
     # (If you need a starting point, the VGG networks work great and are straightforward to use)
    if arch.lower() == "densenet121":
        input_size = 1024
        model = models.densenet121(pretrained=True)
    else:
        input_size = 25088
        model = models.vgg13(pretrained=True)
             
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    #2) Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
    output_size = 102
    model.classifier = nn.Sequential(OrderedDict([
                                 ('hidden1',nn.Linear(input_size, hidden_units)),
                                 ('relu1',nn.ReLU()),
                                 ('hidden_2',nn.Linear(hidden_units, 256)),
                                 ('relu_2',nn.ReLU()),
                                 ('dropout',nn.Dropout(0.2)),
                                 ('output',nn.Linear(256, output_size)),
                                 ('softmax',nn.LogSoftmax(dim=1))]))
    print(f"Model with arch: {arch} and hidden_units {hidden_units} and 256 is built Successfully.")    
    return model

def train_model (model, trainloader, validloader, learning_rate, epochs, gpu):    
    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    #3) Train the classifier layers using backpropagation using the pre-trained network to get the features
    # TODO: Build and train your network
    # Use GPU if it's available
    device = torch.device("cuda" if gpu else "cpu")
    model.to(device)
    
    print_every = 5
    steps = 0
    running_loss = 0
    train_losses, valid_losses = [], []
    with active_session():
        for epoch in range(epochs):
            running_loss = 0
            for inputs, labels in trainloader:
                steps += 1
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)        
                optimizer.zero_grad()
        
                #Forward pass
                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                #Backward pass
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
        
                if steps % print_every == 0:
                    valid_loss = 0
                    accuracy = 0
                    model.eval()
            
                    # turn off gradients
                    with torch.no_grad():
                        for inputs, labels in validloader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)                   
                            valid_loss += batch_loss.item()
                    
                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    #4) Track the loss and accuracy on the validation set to determine the best hyperparameters 
                    train_losses.append(running_loss/print_every)
                    valid_losses.append(valid_loss/len(validloader))
                    print(f"Epoch {epoch+1}/{epochs}.. "
                        f"Train loss: {running_loss/print_every:.3f}.. "
                        f"validation loss: {valid_loss/len(validloader):.3f}.. "
                        f"validation accuracy: {accuracy/len(validloader):.3f}")
                    running_loss = 0
                    model.train()
                
    print("Training is completed.")
    return model, optimizer, criterion


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
     # TODO: Process a PIL image for use in a PyTorch model  
    # First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. 
    # This can be done with the thumbnail or resize methods. 
    width, height = image.size
    aspect_ratio = width / height
    if aspect_ratio > 1:
        image = image.resize((round(aspect_ratio * 256), 256))
    else:
        image = image.resize((256, round(256 / aspect_ratio)))
   
    # Then you'll need to crop out the center 224x224 portion of the image.
    target = 244
    width, height = image.size
    left = (width - target)/2
    top = (height - target)/2
    right = (width + target)/2
    bottom = (height + target)/2
    image = image.crop((round(left), round(top), round(right), round(bottom)))       
    # Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1.
    np_image = np.array(image) / 255   
    # Normalize the image - subtract the means from each color channel, then divide by the standard deviation.
    np_image = (np_image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])   
    # Reorder dimensions
    np_image = np_image.transpose((2, 0, 1))  
    return np_image

def predict(image_path, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''   
    # TODO: Implement the code to predict the class from an image file
    pil_image = Image.open(image_path)
    np_image = process_image(pil_image)
    
    device = torch.device("cuda" if gpu else "cpu")
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        # Convert from numby to pytorch
        images = torch.from_numpy(np_image)
        images = images.unsqueeze(0)
        images = images.type(torch.FloatTensor)
        
        images = images.to(device) #convert to gpu if it is avaliable
        output = model.forward(images) #forward pass
        ps = torch.exp(output) # get the original probabilities

        probs, indices = torch.topk(ps, topk)
        probs = [float(prob) for prob in probs[0]]
        #convert from these indices to the actual class labels using class_to_idx
        idx_to_class = {v: k for k, v in model.class_to_idx.items()}                
        classes = [idx_to_class[int(index)] for index in indices[0]]
    
    return probs, classes
    
    
    