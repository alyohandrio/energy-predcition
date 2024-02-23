import torch
from torch.utils.data import Dataset
import pandas as pd

class MoleculeDataset(Dataset):
    def __init__(self, filepath, train_transform=None, eval_transform=None):
        super().__init__()
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        df = pd.read_csv(filepath, index_col='id')
        splitted = df['xyz'].str.split("\n").apply(lambda x: list(map(lambda y: y.split(), x[2:])))
        df['C_xyz'] = splitted.apply(lambda x: [list(map(float, y[1:])) for y in x if y[0] == "C"])
        df['H_xyz'] = splitted.apply(lambda x: [list(map(float, y[1:])) for y in x if y[0] == "H"])
        df.drop(columns='xyz', inplace=True)
        self.data = df
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = dict(self.data.iloc[int(idx)])
        data['C_xyz'] = torch.tensor(data['C_xyz'])
        data['H_xyz'] = torch.tensor(data['H_xyz'])
        return data

    def get_with_transform(self, idx, mode):
        data = dict(self.data.iloc[int(idx)])
        data['C_xyz'] = torch.tensor(data['C_xyz'])
        data['H_xyz'] = torch.tensor(data['H_xyz'])
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
