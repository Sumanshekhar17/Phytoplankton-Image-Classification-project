## This is the script where we are experimenting with VGG16 without any additional added layers




import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from pylab import cm

from PIL import Image

import os

# Define your transform
# 224 * 224 because that's the required size of the input in VGG19 model


def is_image_file(filename):
    valid_extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp']
    return any(filename.lower().endswith(ext) for ext in valid_extensions)

Height, Width = 224, 224

transform = transforms.Compose([
    transforms.Resize((Height, Width)),
#     transforms.CenterCrop(224),
    transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #add any other transformations here
])



dataset = datasets.ImageFolder(root='Dataset1/', transform=transform, is_valid_file=is_image_file)

# define the size of the training and validating dataset

train_size = int(0.8*len(dataset))
validate_size = int(len(dataset) - train_size)

train_data, test_data = torch.utils.data.random_split(dataset, [train_size, validate_size])

# create dataloader to load your data
batch_size = 64


train_loader = torch.utils.data.DataLoader(train_data, 
                                           batch_size = batch_size, 
                                           shuffle = True,
                                           num_workers = 4,
                                          )

validate_loader = torch.utils.data.DataLoader(test_data, 
                                              batch_size = batch_size,
                                              shuffle = False,
                                              num_workers = 4,
                                             )

from torchvision.utils import make_grid

import torch
import torchvision.models as models
import torch.nn as nn

vgg16 = models.vgg16(pretrained=True)  # Load the pre-trained VGG16


# Freeze all parameters in the feature extraction part and all but the last classifier layer
for param in vgg16.features.parameters():
    param.requires_grad = False

# Optionally, freeze all classifier layers except the last one
for param in vgg16.classifier.parameters():
    param.requires_grad = False

# Unfreeze the last layer (classifier)
vgg16.classifier[6].requires_grad = True


num_features = vgg16.classifier[6].in_features  # Get the input features of the last layer
vgg16.classifier[6] = nn.Linear(num_features, 10)  # Replace the last layer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg16 = vgg16.to(device)

model = vgg16

# You can check how many cores are available on your system using:
import os
print(os.cpu_count())  # Prints the number of cores in your system

# Then, use this number to set the number of threads
torch.set_num_threads(os.cpu_count())

num_epochs=50
lr=1e-4          #learning rate


# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)


criterion = nn.CrossEntropyLoss()
# Only optimize the parameters of the last layer
optimizer = torch.optim.Adam(model.classifier[6].parameters(), lr=lr)



for epoch in range(num_epochs):  # Training the model for a specified number of epochs
    loss_var = 0  # Variable to accumulate losses over an epoch for averaging
    
    # Training phase
    model.train()  # Set the model to training mode
    for idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)  # Move images to the configured device (GPU or CPU)
        labels = labels.to(device)  # Move labels to the same device

        optimizer.zero_grad()  # Reset gradients for this training step
        scores = model(images)  # Compute output by passing images through the model
        loss = criterion(scores, labels)  # Calculate loss between the model's predictions and actual labels
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights

        loss_var += loss.item()  # Add up the loss for averaging later
        if idx % 64 == 0:  # Log information every 64 batches
            print(f'Epoch [{epoch+1}/{num_epochs}] | Step [{idx+1}/{len(train_loader)}] | Loss: {loss_var/(idx+1):.4f}')

    print(f"Average Loss at epoch {epoch+1}: {loss_var/len(train_loader):.4f}")  # Print average loss after the epoch

    # Validation phase
    model.eval()  # Set the model to evaluation mode
    correct = 0
    samples = 0
    with torch.no_grad():  # No need to track gradients during validation
        for images, labels in validate_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)  # Get the predicted classes (highest output scores)
            correct += (preds == labels).sum().item()  # Count correct predictions
            samples += labels.size(0)  # Count total samples processed

        accuracy = 100 * correct / samples  # Calculate accuracy percentage
        print(f"Validation Accuracy at epoch {epoch+1}: {accuracy:.2f}% | Correct {correct} out of {samples} samples")


# Saving model state dictionary
torch.save(model.state_dict(), 'model_state_dict.pth')

# Optionally, save the whole model (not recommended)
torch.save(model, 'full_model.pth')

# If you want to save the optimizer state as well
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss
}
torch.save(checkpoint, 'model_checkpoint.pth')

    

