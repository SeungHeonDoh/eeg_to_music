import os
import torch
import random
import pandas as pd
import numpy as np
import torchaudio
from torch.utils.data import Dataset
from eeg_to_music.constants import (DEAP_DATASET)

class DEAP_Dataset(Dataset):
    def __init__(self, split, feature_type, label_type):
        self.split = split
        self.feature_type = feature_type
        self.label_type = label_type
        self.annotation = torch.load(os.path.join(DEAP_DATASET, f"{self.feature_type}_data.pt"))
        self.n_fft = int(0.5 * 16000)
        self.win_length = int(0.5 * 16000)
        self.hop_length = int(0.25 * 16000)
        self.n_mels = 96
        self.n_mfcc = 13
        melkwargs= {
            'n_fft': self.n_fft,
            'n_mels': self.n_mels,
            'win_length': self.win_length,
            'hop_length': self.hop_length,
            }
        self.mfcc_fn = torchaudio.transforms.MFCC(n_mfcc = self.n_mfcc, melkwargs = melkwargs )
        if split == 'TRAIN':
            self.track_ids = [i for i in self.annotation.keys() if int(i.split("_")[0]) < 28]
            self.fl = [self.annotation[i] for i in self.track_ids]
            random.shuffle(self.fl)
        elif split == 'VALID':
            self.track_ids = [i for i in self.annotation.keys() if 28 < int(i.split("_")[0]) < 30]
            self.fl = [self.annotation[i] for i in self.track_ids]
            random.shuffle(self.fl)
        elif split == 'TEST':
            self.track_ids = [i for i in self.annotation.keys() if 30 < int(i.split("_")[0])]
            self.fl = [self.annotation[i] for i in self.track_ids]        
        elif split == 'ALL':
            self.track_ids = [i for i in self.annotation.keys()]
            self.fl = [self.annotation[i] for i in self.track_ids]        
            
    def __getitem__(self, index):
        item = self.fl[index]
        track_id = self.track_ids[index]
        eeg = item['feature'].astype(np.float32)
        wav = item['wav'].astype(np.float32)
        mfcc = self.mfcc_fn(torch.from_numpy(wav))
        mfcc = mfcc.mean(dim=-1)
        binary = item[f'{self.label_type}_label'].astype(np.float32)
        return {
            "track_id": track_id,
            "eeg": torch.from_numpy(eeg).unsqueeze(0),
            "wav": mfcc.unsqueeze(0),
            "binary": torch.from_numpy(binary).unsqueeze(0),
        }
        
    def __len__(self):
        return len(self.fl)