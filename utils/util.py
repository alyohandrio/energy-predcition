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
    acc_sum = list(accumulate([0] + lhs))
    starts = [x for i in range(len(lhs)) for xx in range(acc_sum[i], acc_sum[i + 1]) for x in [xx] * rhs[i]]
    acc_sum = list(accumulate([0] + rhs))
    ends = [x for i in range(len(lhs)) for x in list(range(acc_sum[i], acc_sum[i + 1])) * lhs[i]]
    return torch.tensor([starts, ends])


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

