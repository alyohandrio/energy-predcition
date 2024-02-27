import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class MoleculeDataset(Dataset):
    def __init__(self, filepath, train_transform=None, eval_transform=None):
        super().__init__()
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        df = pd.read_csv(filepath, index_col='id')
        splitted = df['xyz'].str.split("\n").apply(lambda x: list(map(lambda y: y.split(), x[2:])))
        df['C_xyz'] = splitted.apply(lambda x: [list(map(float, y[1:])) for y in x if y[0] == "C"])
        df['H_xyz'] = splitted.apply(lambda x: [list(map(float, y[1:])) for y in x if y[0] == "H"])

        C_lens = df['C_xyz'].apply(lambda x: len(x)).values
        H_lens = df['H_xyz'].apply(lambda x: len(x)).values
        self.C_starts = np.concatenate([[0], C_lens]).cumsum()
        self.H_starts = np.concatenate([[0], H_lens]).cumsum()

        self.C_xyz = torch.from_numpy(np.concatenate(df['C_xyz'].values)).float()
        self.H_xyz = torch.from_numpy(np.concatenate(df['H_xyz'].values)).float()
        self.U_0 = torch.from_numpy(df['U_0'].values).float()

    def __len__(self):
        return len(self.U_0)

    def __getitem__(self, idx):
        data = {}
        data['C_xyz'] = self.C_xyz[self.C_starts[idx]:self.C_starts[idx + 1]]
        data['H_xyz'] = self.H_xyz[self.H_starts[idx]:self.H_starts[idx + 1]]
        data['U_0'] = self.U_0[idx]
        return data

    def get_with_transform(self, idx, mode):
        data = {}
        data['C_xyz'] = self.C_xyz[self.C_starts[idx]:self.C_starts[idx + 1]]
        data['H_xyz'] = self.H_xyz[self.H_starts[idx]:self.H_starts[idx + 1]]
        data['U_0'] = self.U_0[idx]
        if mode == "train":
            transform = self.train_transform
        elif mode == "eval":
            transform = self.eval_transform
        if transform is not None:
            x = torch.cat([data['C_xyz'], data['H_xyz']])
            x = transform(x)
            data['H_xyz'] = x[len(data['C_xyz']):]
            data['C_xyz'] = x[:len(data['C_xyz'])]

        return data
