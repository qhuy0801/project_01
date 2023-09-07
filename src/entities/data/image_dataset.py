from abc import ABC
import random

import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset, ABC):
    dataset_dir: str
    target_size: int
    data: []
    class_dict: {}

    def __len__(self):
        return len(self.data)

    def get_sample_labels(self, sample_num):
        """
        Get random number of samples
        :param sample_num:
        :return:
        """
        if len(self.class_dict) < sample_num:
            raise ValueError(
                f"You've requested to get random {sample_num} classes while there are only "
                f"{len(self.class_dict)} classes in the dataset"
            )
        return torch.tensor(random.sample(list(self.class_dict.values()), sample_num))
