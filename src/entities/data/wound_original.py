import os

import cv2
import torch
from torch.utils.data.dataset import T_co

from entities.data.image_dataset import ImageDataset
from utils import forward_transform


class WoundOriginal(ImageDataset):
    """
    Original dataset for training the autoencoder
    """

    def __init__(self, dataset_dir: str = "../data/images/", target_size: int = 512) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.target_size = target_size

        self.data = []

        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.endswith(('jpg', 'jpeg')):
                    self.data.append(os.path.join(root, file))

    def __getitem__(self, index) -> T_co:
        file_path = self.data[index]
        image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
        return forward_transform(
            image=image, target_size=self.target_size
        ), file_path
