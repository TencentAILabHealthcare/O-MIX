from typing import Iterable, List, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler, SubsetRandomSampler, BatchSampler


class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices: Sequence[int]):
        self.indices = indices

    def __iter__(self) -> Iterable[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class SubsetsBatchSampler(Sampler[List[int]]):
    r"""Samples batches of indices from a list of subsets of indices. Each subset
    of indices represents a data subset and is sampled without replacement randomly
    or sequentially. Specially, each batch only contains indices from a single subset.
    This sampler is for the scenario where samples need to be drawn from multiple
    subsets separately.

    Arguments:
        subsets (List[Sequence[int]]): A list of subsets of indices.
        batch_size (int): Size of mini-batch.
        intra_subset_shuffle (bool): If ``True``, the sampler will shuffle the indices
            within each subset.
        inter_subset_shuffle (bool): If ``True``, the sampler will shuffle the order
            of subsets.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
    """

    def __init__(
        self,
        subsets: List[Sequence[int]],
        batch_size: int,
        intra_subset_shuffle: bool = True,
        inter_subset_shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.subsets = subsets
        self.batch_size = batch_size
        self.intra_subset_shuffle = intra_subset_shuffle
        self.inter_subset_shuffle = inter_subset_shuffle
        self.drop_last = drop_last

        if intra_subset_shuffle:
            self.subset_samplers = [SubsetRandomSampler(subset) for subset in subsets]
        else:
            self.subset_samplers = [
                SubsetSequentialSampler(subset) for subset in subsets
            ]

        self.batch_samplers = [
            BatchSampler(sampler, batch_size, drop_last)
            for sampler in self.subset_samplers
        ]

        if inter_subset_shuffle:
            # maintain a mapping from sample batch index to batch sampler
            _id_to_batch_sampler = []
            for i, batch_sampler in enumerate(self.batch_samplers):
                _id_to_batch_sampler.extend([i] * len(batch_sampler))
            self._id_to_batch_sampler = np.array(_id_to_batch_sampler)

            assert len(self._id_to_batch_sampler) == len(self)

            self.batch_sampler_iterrators = [
                batch_sampler.__iter__() for batch_sampler in self.batch_samplers
            ]

    def __iter__(self) -> Iterable[List[int]]:
        if self.inter_subset_shuffle:
            # randomly sample from batch samplers
            random_idx = torch.randperm(len(self._id_to_batch_sampler))
            batch_sampler_ids = self._id_to_batch_sampler[random_idx]
            for batch_sampler_id in batch_sampler_ids:
                batch_sampler_iter = self.batch_sampler_iterrators[batch_sampler_id]
                yield next(batch_sampler_iter)
        else:
            for batch_sampler in self.batch_samplers:
                yield from batch_sampler

    def __len__(self) -> int:
        return sum(len(batch_sampler) for batch_sampler in self.batch_samplers)

import torch.distributed as dist
import math
class DistributedSubsetsBatchSampler(Sampler[List[int]]):
    """
    A Distributed version of SubsetsBatchSampler.
    It wraps a SubsetsBatchSampler and distributes the generated batches
    among the replicas.

    Arguments:
        subsets (List[Sequence[int]]): A list of subsets of indices.
        batch_size (int): Size of mini-batch.
        intra_subset_shuffle (bool): If ``True``, shuffle indices within each subset.
        inter_subset_shuffle (bool): If ``True``, shuffle the order of batches across subsets.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, `dist.get_world_size()` is used.
        rank (int, optional): Rank of the current process in `num_replicas`.
            By default, `dist.get_rank()` is used.
        seed (int): A random seed to ensure that the shuffling of batches is identical
            across all processes.
    """

    def __init__(
            self,
            subsets: List[Sequence[int]],
            batch_size: int,
            intra_subset_shuffle: bool = True,
            inter_subset_shuffle: bool = True,
            num_replicas: int = None,
            rank: int = None,
            seed: int = 42,
    ):
        if num_replicas is None:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("Requires distributed package to be initialized")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("Requires distributed package to be initialized")
            rank = dist.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        # This base sampler is used to generate the full list of batches on each process.
        # drop_last is False because we will handle uneven division ourselves.
        self.base_sampler = SubsetsBatchSampler(
            subsets, batch_size, intra_subset_shuffle, inter_subset_shuffle, drop_last=False
        )

        self.total_num_batches = len(self.base_sampler)

        # Calculate the number of batches for this specific replica
        self.num_batches_per_replica = math.ceil(self.total_num_batches / self.num_replicas)

        # The total size needs to be padded to be divisible by the number of replicas
        self.padded_total_size = self.num_batches_per_replica * self.num_replicas

    def __iter__(self) -> Iterable[List[int]]:
        # 1. Generate the full list of batches.
        # It's crucial that this list is the same on all processes BEFORE shuffling.
        # The internal shuffling of SubsetsBatchSampler (intra_subset_shuffle) is okay,
        # as we will deterministically shuffle the resulting BATCHES.
        all_batches = list(self.base_sampler)

        # 2. Deterministically shuffle the order of batches.
        # All processes will have the same shuffled order because they use the same seed.
        if self.base_sampler.inter_subset_shuffle:
            g = torch.Generator()
            # Use epoch and seed to get a different shuffle for each epoch
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in indices]

        # 3. Add padding to make the total number of batches divisible by num_replicas.
        # This is to ensure all processes have the same number of batches.
        padding_size = self.padded_total_size - len(all_batches)
        if padding_size > 0:
            all_batches += all_batches[:padding_size]

        # Sanity check
        assert len(all_batches) == self.padded_total_size

        # 4. Subsample the batches for the current rank.
        my_batches = all_batches[self.rank: self.padded_total_size: self.num_replicas]

        return iter(my_batches)

    def __len__(self) -> int:
        return self.num_batches_per_replica

    def set_epoch(self, epoch: int) -> None:
        """
        Sets the epoch for this sampler. When `inter_subset_shuffle=True`, this ensures a
        different random permutation of batches for each epoch.
        """
        self.epoch = epoch