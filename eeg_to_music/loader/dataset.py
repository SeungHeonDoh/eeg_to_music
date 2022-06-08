import os
import torch
import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from eeg_to_music.constants import (DEAP_DATASET)

class DEAP_Dataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.annotation = torch.load(os.path.join(DEAP_DATASET, f"annotation.pt"))
        if split == 'TRAIN':
            self.fl = [self.annotation[i] for i in self.annotation.keys() if int(i.split("_")[0]) < 28]
        elif split == 'VALID':
            self.fl = [self.annotation[i] for i in self.annotation.keys() if 28 < int(i.split("_")[0]) < 30]
        elif split == 'TEST':
            self.fl = [self.annotation[i] for i in self.annotation.keys() if 30 < int(i.split("_")[0])]
            
    def __getitem__(self, index):
        item = self.fl[index]
        data = item['data'].astype(np.float32)
        label = item['label'].astype(np.float32)
        subject = item['subject']
        trial = item['trial']
        return torch.from_numpy(data), torch.from_numpy(label), subject, trial
        
    def __len__(self):
        return len(self.fl)