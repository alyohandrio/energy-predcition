import torch
from torch_geometric.data import Data, HeteroData
from utils import create_edges


def hetero_collate_fn(batch):
    res = HeteroData()
    num_Cs = torch.tensor([len(x['C_xyz']) for x in batch])
    num_Hs = torch.tensor([len(x['H_xyz']) for x in batch])
    U_0 = torch.tensor([x['U_0'] for x in batch])

    res['C'].x = torch.cat([x['C_xyz'] for x in batch])
    res['H'].x = torch.cat([x['H_xyz'] for x in batch])
    res['C', 'CC', 'C'].edge_index = create_edges(num_Cs, num_Cs)
    res['C', 'CH', 'H'].edge_index = create_edges(num_Cs, num_Hs)
    res['H', 'HC', 'C'].edge_index = create_edges(num_Hs, num_Cs)
    res['H', 'HH', 'H'].edge_index = create_edges(num_Hs, num_Hs)
    C_group = torch.arange(len(num_Cs)).repeat_interleave(num_Cs)
    H_group = torch.arange(len(num_Hs)).repeat_interleave(num_Hs)
    return {"data": res, "C_group": C_group, "H_group": H_group, "U_0": U_0.float(), 'num': len(batch)}


def homo_collate_fn(batch):
    res = Data()
    num_Cs = torch.tensor([len(x['C_xyz']) for x in batch])
    num_Hs = torch.tensor([len(x['H_xyz']) for x in batch])
    nums = num_Cs + num_Hs
    U_0 = torch.tensor([x['U_0'] for x in batch])
    
    res.x = torch.cat([torch.cat([torch.cat([x['C_xyz'], torch.zeros(len(x['C_xyz']), 1)], dim=-1), torch.cat([x['H_xyz'], torch.ones(len(x['H_xyz']), 1)], dim=-1)]) for x in batch])
    res.edge_index = create_edges(nums, nums)
    group = torch.arange(len(nums)).repeat_interleave(nums)
    return {"data": res, "group": group, "U_0": U_0.float(), 'num': len(batch)}
