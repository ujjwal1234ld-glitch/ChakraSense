import numpy as np
import pandas as pd
import torch
from torch import nn
from pathlib import Path

class CycloneNet(nn.Module):
    """Small two-channel encoder and causal GRU. No external pretrained weights."""
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(2,16,5,stride=2,padding=2),nn.GroupNorm(4,16),nn.SiLU(),
            nn.Conv2d(16,32,3,stride=2,padding=1),nn.GroupNorm(8,32),nn.SiLU(),
            nn.Conv2d(32,64,3,stride=2,padding=1),nn.GroupNorm(8,64),nn.SiLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.temporal=nn.GRU(64,48,batch_first=True)
        self.wind=nn.Linear(48,1);self.ri=nn.Linear(48,1)
    def forward(self,x):
        b,t,c,h,w=x.shape
        z=self.encoder(x.reshape(b*t,c,h,w)).reshape(b,t,-1)
        z,_=self.temporal(z);last=z[:,-1]
        return {'wind':self.wind(last).squeeze(-1),'logit':self.ri(last).squeeze(-1)}

def normalise(bt):
    # Fixed physical scaling, identical in training and inference; no test-fitted stats.
    return ((bt.astype(np.float32)-250.)/50.).astype(np.float32)

class StormDataset(torch.utils.data.Dataset):
    def __init__(self,manifest,split):
        self.root=Path(manifest).parent
        self.rows=pd.read_csv(manifest);self.rows=self.rows[self.rows.split==split].reset_index(drop=True)
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows.iloc[i]
        with np.load(self.root/r.path,allow_pickle=False) as z:x=z['bt']
        return torch.from_numpy(normalise(x)),torch.tensor(float(r.wind_kt)/100.),torch.tensor(float(r.ri))
