import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from scipy import signal


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", apply_preprocessing: bool = True) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."
        
        # if apply_preprocessing:
        #     self.X = self.preprocess(self.X)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
    
    def normalize(self, X):
        mean = X.mean(dim=2, keepdim=True)
        std = X.std(dim=2, keepdim=True)
        return (X - mean) / (std + 1e-8)
    
    def bandpass_filter(self, X, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.filtfilt(b, a, X.numpy(), axis=2)
    
    def baseline_correction(self, X, baseline_period=200):
        baseline = X[:, :, :baseline_period].mean(dim=2, keepdim=True)
        return X - baseline

    def preprocess(self, X):
        X = self.normalize(X)
        # X = torch.tensor(self.bandpass_filter(X, lowcut=0.5, highcut=100, fs=200))
        # X = self.baseline_correction(X)
        # X = self.resample(X, original_fs=200, target_fs=100)  # If resampling is needed
        return X
