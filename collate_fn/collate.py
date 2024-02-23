import torch
from torch_geometric.data import Data, HeteroData
from utils import create_edges

def hetero_collate_fn(batch):
    res = HeteroData()
    num_Cs = [len(x['C_xyz']) for x in batch]
    num_Hs = [len(x['H_xyz']) for x in batch]
    # sums = torch.cat([torch.tensor(x['C_xyz']).sum(dim=0, keepdim=True) + torch.tensor(x['H_xyz']).sum(dim=0, keepdim=True) for x in batch])
    # means = sums / (torch.tensor(num_Cs) + torch.tensor(num_Hs))[..., None]
    U_0 = torch.tensor([x['U_0'] for x in batch])
    
    # res['C'].x = torch.cat([torch.tensor(x['C_xyz']) - avg for x, avg in zip(batch, means)])
    # res['H'].x = torch.cat([torch.tensor(x['H_xyz']) - avg for x, avg in zip(batch, means)])
    res['C'].x = torch.cat([x['C_xyz'] for x in batch])
    res['H'].x = torch.cat([x['H_xyz'] for x in batch])
    res['C', 'CC', 'C'].edge_index = create_edges(num_Cs, num_Cs)
    res['C', 'CH', 'H'].edge_index = create_edges(num_Cs, num_Hs)
    res['H', 'HC', 'C'].edge_index = create_edges(num_Hs, num_Cs)
    res['H', 'HH', 'H'].edge_index = create_edges(num_Hs, num_Hs)
    C_group = torch.tensor([j for i in range(len(num_Cs)) for j in [i] * num_Cs[i]])
    H_group = torch.tensor([j for i in range(len(num_Cs)) for j in [i] * num_Hs[i]])
    return {"data": res, "C_group": C_group, "H_group": H_group, "U_0": U_0.float(), 'num': len(batch)}

def homo_collate_fn(batch):
    res = Data()
    num_Cs = [len(x['C_xyz']) for x in batch]
    num_Hs = [len(x['H_xyz']) for x in batch]
    nums = [x + y for x, y in zip(num_Cs, num_Hs)]
    U_0 = torch.tensor([x['U_0'] for x in batch])
    
    tmp_x = torch.cat([torch.cat([x['C_xyz'], x['H_xyz']]) for x in batch])
    atoms = torch.tensor([v for x, y in zip(num_Cs, num_Hs) for v in [0] * x + [1] * y])[:, None]
    tmp_x = torch.cat([tmp_x, atoms], dim=-1)
    res.x = tmp_x
    res.edge_index = create_edges(nums, nums)
    group = torch.tensor([j for i in range(len(nums)) for j in [i] * nums[i]])
    return {"data": res, "group": group, "U_0": U_0.float(), 'num': len(batch)}