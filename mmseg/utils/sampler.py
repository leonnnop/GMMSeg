from torch.utils.data import Dataset, Sampler
from torch.utils.data import DistributedSampler
from operator import itemgetter
from typing import Iterator, List, Optional, Union
import numpy as np
import torch
from torch.nn import functional as F
import random
import math
from .logger import get_root_logger

class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class BatchBalanceClassSampler(Sampler):

    def __init__(
        self,
        dataset,
        cfg
    ):
        """Sampler initialisation."""
        super().__init__(dataset)
        logger = get_root_logger()
        logger.info(f'Enable Sampling Mode [BALANCE] ')
        self._num_classes = 1 # * one class per batch
        self._batch_size = 1
        self._num_batches = len(dataset) // self._batch_size
        self._labels, self.lbl2idx = self.gather_labels(dataset)
        
    def gather_labels(self, dataset):
        num_labels = len(dataset.CLASSES)
        labels = list(range(num_labels))

        _dataset_dict = {
            # * only compatible with ADE20K, Cityscapes and COCO-Stuff
            # TODO hard-coded: ugly, may fix
            150: 'mmseg/utils/sampler/ade_lbl2idx.pth',
            19:  'mmseg/utils/sampler/city_lbl2idx.pth',
            171: 'mmseg/utils/sampler/cocos_lbl2idx.pth',
        }

        lbl2idx = torch.load(_dataset_dict[num_labels])

        return labels, lbl2idx

    def __len__(self) -> int:
        """
        Returns:
            number of samples in an epoch
        """
        return self._num_batches

    def __iter__(self) -> Iterator[int]:
        """
        Returns:
            indeces for sampling dataset elems during an epoch
        """
        indices = []
        for _ in range(self._num_batches):
            cls_id = random.sample(self._labels, self._num_classes)[0]
            replace_flag = self._batch_size > len(self.lbl2idx[cls_id])
            batch_indices = np.random.choice(
                self.lbl2idx[cls_id], self._batch_size, replace=replace_flag
            ).tolist()
            indices.append(batch_indices[0])

        return iter(indices)



class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.
        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


def build_sampler(dataset, world_size, rank, shuffle, cfg):
    if cfg.model.train_cfg.get('sampler_mode', None) is None:
        return DistributedSampler(dataset, world_size, rank, shuffle=shuffle)
    elif str(cfg.model.train_cfg.sampler_mode).lower() == 'gmmseg':
        sampler = BatchBalanceClassSampler(dataset, cfg=cfg)
        return DistributedSamplerWrapper(sampler, world_size, rank, shuffle=shuffle)
