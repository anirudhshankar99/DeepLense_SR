import torch
import numpy as np
import pandas as pd


num_samples = {'vanilla':1000, 'sparse':300}
class LensingDataset(torch.utils.data.Dataset):
    def __init__(self, directory, sample, transform=None):
        super(LensingDataset, self).__init__()
        self.directory = directory
        self.transform = transform
        self.sample = sample
    def __len__(self):
        return num_samples[self.sample]
    
    def __getitem__(self, index):
        if self.sample == 'vanilla':
            img_path = self.directory + 'LR/' + 'sample%d.npy'%(index+1)
            label_path =  self.directory + 'HR/' + 'sample%d.npy'%(index+1)
        else:
            img_path = self.directory + 'LR/' + 'LR_%d.npy'%(index+1)
            label_path =  self.directory + 'HR/' + 'HR_%d.npy'%(index+1)
        image = torch.tensor(np.load(img_path))
        label = torch.tensor(np.load(label_path))
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        return (image,label)