import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
from skimage import io
import skimage
from medmnist import PathMNIST

class ImageDataset(Dataset):
    def __init__(self, data, label, transform):
        self.data = data
        self.label = label
        self.transform = transform
        
    def __getitem__(self, index):
        img_path = self.data[index]
        img = Image.open(img_path)
        img = img.convert(mode='RGB')
        label = torch.ones(1, dtype=torch.long) * self.label[index]
        img = self.transform(img)
        return img_path, img, label
    
    def __len__(self):
        return len(self.data)

class PathMNISTDataset(PathMNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        img_path = f"PathMNIST_{index}"
        img, label = super().__getitem__(index)
        return img_path, img, label
