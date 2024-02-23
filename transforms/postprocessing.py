import torch

class AddNorm:
    def __init__(self, post=None):
        self.post = post

    def __call__(self, x):
        norms = x[:, :3].norm(dim=-1, keepdim=True)
        if self.post is not None:
            norms = self.post(norms)
        return torch.cat([x, norms], dim=-1)

class NormDirection:
    def __call__(self, x):
        norms = x[:, :3].norm(dim=-1, keepdim=True)
        return torch.cat([x[:, :3] / norms, x[:, 3:]], dim=-1)

