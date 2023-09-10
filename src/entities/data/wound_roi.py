import glob

import cv2
import torch
from torch.utils.data.dataset import T_co

from entities.data.image_dataset import ImageDataset
from utils import forward_transform


class WoundROI(ImageDataset):
    """
    Wound ROI dataset for training of the pattern
    """

    def __init__(
        self,
        dataset_dir: str = "../../data/segmented/roi/",
        target_size: int = 512,
    ) -> None:
        super().__init__()

        self.dataset_dir = dataset_dir
        self.target_size = target_size
        self.__class_tuple = ()

        __file_list = glob.glob(self.dataset_dir + "*")
        for __class_path in __file_list:
            __class_name = __class_path.split("/")[-1]
            self.__class_tuple = (*self.__class_tuple, __class_name)
            for __img_path in glob.glob(__class_path + "/*"):
                self.data.append([__img_path, __class_name])

        self.class_dict = dict(zip(self.__class_tuple, range(len(self.__class_tuple))))

    def __getitem__(self, index) -> T_co:
        image_path, label = self.data[index]
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        return forward_transform(
            image=image, target_size=self.target_size
        ), torch.tensor(self.class_dict[label])
