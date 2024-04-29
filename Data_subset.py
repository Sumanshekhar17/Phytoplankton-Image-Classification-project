import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from pylab import cm

from PIL import Image

# we are using IOmageFolder to see how our dataset looks like in proper labeled manner. This is why we are not using any transform right now.
dataset = datasets.ImageFolder(root='./DeepLearningProject/Pythoplankton_image_Data/', transform=None)

# store all the label corresponding to the classes
classes = dataset.class_to_idx

# let's see how many images each label have and store it in label_count variable for future use.

import os

#common root path to all the classes
data_path = './DeepLearningProject/Pythoplankton_image_Data'

label_counts = {}


for j, label in enumerate(os.listdir(data_path)):
    
    if os.path.isdir(os.path.join(data_path, label)):
        # Count the number of files in each label's directory
        label_counts[label] = len(os.listdir(os.path.join(data_path, label)))
        
        
# trying to remove dataset with image greater than 600 and less than 500 

keys1 = []
keys2 = []
keys3 = []
keys4 = []

for key, values in label_counts.items():
    if 300 < values <= 500:
        keys1.append(key)
    elif 500 < values <= 700:
        keys2.append(key)
    elif 700 < values < 1415:
        keys3.append(key)
    elif 7000 < values :
        keys4.append(key)
        
        
#Defining Custom Dataset class for creating selected class dataset according to our need

class CustomPhytoplanktonDataset(Dataset):
    
    def __init__(self, folder_paths, transform=None):
        """
        Args:
            folder_paths (list): List of strings with paths to folders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.folder_paths = folder_paths
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_name = []

        self._load_dataset()


                    
    def _load_dataset(self):
        # Supported image file extensions
        image_extensions = {'.jpg', '.jpeg', '.png'}

        # Iterate through all the provided folder paths
        for idx, folder_path in enumerate(self.folder_paths):
            
            # Assuming folder names are the class labels
            
            class_name = os.path.basename(folder_path)
            
            for image_file in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_file)
                _, ext = os.path.splitext(image_path)  # Extract the file extension
                if os.path.isfile(image_path) and ext.lower() in image_extensions:
                    # Store the image path and its class label
                    self.images.append(image_path)
                    self.labels.append(idx)  # idx, or class name if you prefer numerical labels
                    self.class_name.append(class_name)
               

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        # Load image and label from the stored path and label
        
        image_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')  # Convert to RGB to ensure 3 channels

        if self.transform:
            image = self.transform(image)
        return image, label
            
    def __repr__(self):
        format_string = 'Custom Dataset \n'
        format_string += '    Number of datapoints: {}\n'.format(self.__len__())
        format_string += '    Name of classes: {}\n'.format(set(self.class_name))
        format_string += '    Root location: {}\n'.format(os.path.commonpath(self.folder_paths))
        format_string += '    StandardTransform\n'
        format_string += 'Transform: {}\n'.format(self.transform.__repr__().replace('\n', '\n    '))
        return format_string
    
    
# Define your transform
# 224 * 224 because that's the required size of the input in VGG19 model

Height, Width = 224, 224

transform = transforms.Compose([
    transforms.Resize((Height, Width)),
    transforms.ToTensor(),
    # Add any other transformations here
])

# List of folder paths
folder_paths1 = []
for key in keys1:
    folder_paths1.append(data_path + '/' + key)
    
    
folder_paths2 = []
for key in keys2:
    folder_paths2.append(data_path + '/' + key)
    
folder_paths3 = []
for key in keys3:
    folder_paths3.append(data_path + '/' + key)
    

    
# Create an instance of your custom dataset
dataset1 = CustomPhytoplanktonDataset(folder_paths1, transform = transform)

dataset2 = CustomPhytoplanktonDataset(folder_paths2, transform = transform)

dataset3 = CustomPhytoplanktonDataset(folder_paths3, transform = transform)