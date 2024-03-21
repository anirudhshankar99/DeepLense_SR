import torch
import numpy as np
import pandas as pd


num_samples = 1000
class LensingDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform=None):
        super(LensingDataset, self).__init__()
        self.directory = directory
        self.transform = transform
    
    def __len__(self):
        return num_samples
    
    def __getitem__(self, index):
        img_path = self.directory + 'LR/' + 'sample%d.npy'%(index+1)
        label_path =  self.directory + 'HR/' + 'sample%d.npy'%(index+1)
        image = torch.tensor(np.load(img_path))
        label = torch.tensor(np.load(label_path))
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        return (image,label)