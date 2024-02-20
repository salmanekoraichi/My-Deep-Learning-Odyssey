
# ------------------------------------------------------------------
#     _____ _     _ _
#    |  ___(_) __| | | ___
#    | |_  | |/ _` | |/ _ \
#    |  _| | | (_| | |  __/
#    |_|   |_|\__,_|_|\___|
# ------------------------------------------------------------------
# Formation Introduction au Deep Learning  (FIDLE)
# CNRS/SARI/DEVLOG 2023 
# ------------------------------------------------------------------
# 2.0 version by Achille Mbogol Touye (EFELIA-MIAI/SIMAP¨), sep 2023

import torch
import pandas as pd
import lightning.pytorch as pl


class WineQualityDataset(pl.LightningDataModule):
    """Wine Quality dataset."""

    def __init__(self, csv_file, transform=None):
        
        """
        Args:
            csv_file (string): Path to the csv file.
            
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__()
        self.csv_file=csv_file
        self.data = pd.read_csv(self.csv_file, header=0, sep=';')
        self.transform = transform

    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        features = self.data.iloc[idx, :-1].values.astype('float32')
        target   = self.data.iloc[idx, -1:].values.astype('float32')
        sample = {'features':features, 'quality':target}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
        

class Normalize(WineQualityDataset):
    """normalize data"""
    def __init__(self, csv_file):
        mean,std=self.compute_mean_and_std(csv_file)
        self.mean=mean
        self.std=std

    def compute_mean_and_std(self, csv_file):
        """Compute the mean and std for each feature."""
        dataset= WineQualityDataset(csv_file)
        mean   = dataset.data.iloc[:,:-1].mean(axis=0).values.astype('float32')
        std    = dataset.data.iloc[:,:-1].std(axis=0).values.astype('float32')
        return mean,std    
        
    
    def __call__(self, sample):
        features, target = sample['features'],sample['quality']
        norm_features = (features - self.mean) / self.std     # normalize features
        return {'features':norm_features,
                 'quality':target
               }

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        features, target = sample['features'], sample['quality']        
        return {'features': torch.from_numpy(features),
                'quality' : torch.from_numpy(target)
               }    


