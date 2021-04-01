import glob, os, sys
import pandas as pd
import numpy as np

from PIL import Image
from functools import partial
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# sys.path.insert(0, os.path.join(os.path.expanduser('~/Research/Github/'),'PyTorch-VAE'))

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
    
class WCDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        
        self.data_paths = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        

    def __len__(self):
        return(len(self.data_paths))
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,self.data_paths.iloc[idx,1],self.data_paths.iloc[idx,2])
        sample = pil_loader(img_name)
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

class WCShotgunDataset(Dataset):
    def __init__(self, csv_file, N_fm, root_dir, transform=None):
        
        self.data_paths = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.N_fm = N_fm

    def __len__(self):
        return(len(self.data_paths))
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample=[]
        for n in range(self.N_fm):
            img_name = eval(self.data_paths['FileName'][idx])[n]
            img_path = os.path.join(self.root_dir,self.data_paths.iloc[idx,1],img_name)
            img = pil_loader(img_path)
            if self.transform:
                img = self.transform(img)
            sample.append(img)
        sample = torch.cat(sample,dim=0)
        return sample

class WC3dDataset(Dataset):
    def __init__(self, csv_file, N_fm, root_dir, transform=None):
        
        self.data_paths = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.N_fm = N_fm

    def __len__(self):
        return(len(self.data_paths))
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample=[]
        for n in range(self.N_fm):
            img_name = eval(self.data_paths['FileName'][idx])[n]
            img_path = os.path.join(self.root_dir,self.data_paths.iloc[idx,1],img_name)
            img = pil_loader(img_path)
            if self.transform:
                img = self.transform(img)
            sample.append(img)
        sample = torch.cat(sample,dim=0).unsqueeze(1) # To put into B x C x D x W x H format
        return sample