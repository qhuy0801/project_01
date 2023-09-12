import os

import cv2
import torch
from torch.utils.data.dataset import T_co
from skimage import io

from entities.data.image_dataset import ImageDataset
from utils import forward_transform, get_metadata, find_path


class WoundOriginal(ImageDataset):
    """
    Original dataset for training the autoencoder
    """

    def __init__(
        self,
        metadata_path: str = "./data/annotations/merged_wound_details_july_2022.csv",
        dataset_dir: str = "./data/original/",
        target_size: int = 512,
    ) -> None:
        super().__init__()
        self.data = []
        self.metadata = get_metadata(metadata_path, 0)
        self.target_size = target_size

        self.__class_tuple = ()

        for _, row in self.metadata.iterrows():
            wound_img_path = find_path(dataset_dir, row["ImageDetail.ImageFilename"])
            if wound_img_path is not None:
                self.__class_tuple = (*self.__class_tuple, row["ImageDetail.WoundType"])
                self.data.append([wound_img_path, row["ImageDetail.WoundType"]])

        self.class_dict = dict(zip(self.__class_tuple, range(len(self.__class_tuple))))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> T_co:
        wound_img_path, label = self.data[index]
        wound_img = cv2.cvtColor(cv2.imread(wound_img_path), cv2.COLOR_BGR2RGB)
        return forward_transform(wound_img, self.target_size), torch.tensor(self.class_dict[label])
