import torch

class Rotation:
    def __call__(self, x):
        alpha, beta, gamma = torch.rand(3) * 2 * torch.pi
        m1 = torch.tensor([[1, 0, 0], [0, alpha.cos(), -alpha.sin()], [0, alpha.sin(), alpha.cos()]]).to(x.device)
        m2 = torch.tensor([[beta.cos(), 0, beta.sin()], [0, 1, 0], [-beta.sin(), 0, beta.cos()]]).to(x.device)
        m3 = torch.tensor([[gamma.cos(), -gamma.sin(), 0], [gamma.sin(), gamma.cos(), 0], [0, 0, 1]]).to(x.device)
        return torch.cat([x[:, :3] @ (m1 @ m2 @ m3), x[:, 3:]], dim=-1)

class Center:
    def __call__(self, x):
        center = x[:, :3].mean(dim=0, keepdim=True)
        return torch.cat([x[:, :3] - center, x[:, 3:]], dim=-1)
