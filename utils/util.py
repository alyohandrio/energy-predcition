from itertools import accumulate
import math
import torch


class SubsetWithTransform(torch.utils.data.Dataset):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.mode = None

    def train(self):
        self.mode = "train"

    def eval(self):
        self.mode = "eval"

    def __getitem__(self, idx):
        return self.dataset.get_with_transform(self.indices[idx], self.mode)

    def __len__(self):
        return len(self.indices)

def create_edges(lhs, rhs):
    sizes = lhs * rhs
    a = torch.arange(lhs.sum()).repeat_interleave(rhs.repeat_interleave(lhs))
    tmp = torch.arange(sizes.sum()) - torch.cat([torch.zeros(1).int(), sizes])[:-1].cumsum(0).repeat_interleave(sizes)
    b = tmp % rhs.repeat_interleave(sizes) + torch.cat([torch.zeros(1).int(), rhs])[:-1].cumsum(0).repeat_interleave(sizes)
    return torch.vstack([a, b])

def random_split(dataset, lengths, random_state=None):
    generator = torch.Generator().manual_seed(random_state)
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. "
                              f"This might result in an empty dataset.")

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = torch.randperm(sum(lengths), generator=generator).tolist()
    return [SubsetWithTransform(dataset, indices[offset - length : offset]) for offset, length in zip(accumulate(lengths), lengths)]

