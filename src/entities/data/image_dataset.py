from abc import ABC

from torch.utils.data import Dataset
import albumentations as A


class ImageDataset(Dataset, ABC):
    dataset_dir: str
    target_size: int
    data: []
    class_dict: {}

    def __len__(self):
        return len(self.data)
