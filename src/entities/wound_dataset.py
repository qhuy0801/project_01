import glob
import os.path
import warnings

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

import CONST
from utils import forward_transform


class WoundDataset(Dataset):
    def __init__(
        self,
        image_dir: str = CONST.PROCESSED_IMAGES_DIR,
        segment_dir: str = CONST.PROCESSED_SEGMENT_DIR,
        annotation_path: str = CONST.ANNOTATION_PROCESSED_PATH,
        target_tensor_size: int = 512,
    ) -> None:
        super().__init__()
        self.image_dir = image_dir
        self.segment_dir = segment_dir
        self.annotation_path = annotation_path
        self.target_tensor_size = target_tensor_size

        self.data = []

        self.annotation = pd.read_csv(self.annotation_path)

        self.data = [
            *self.data,
            *glob.glob(f"{self.image_dir}*.jpeg"),
            *glob.glob(f"{self.image_dir}*.jpg"),
            *glob.glob(f"{self.image_dir}*.png"),
        ]

        if len(self.data) == 0:
            warnings.warn(
                f"Dataset is empty, check the directory path again"
                f"Current directory path: {self.image_dir}"
            )

    def __getitem__(self, index) -> T_co:
        file_path = self.data[index]
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
        segment_path = glob.glob(f"{self.segment_dir}{file_name}*")[0]
        return (
            forward_transform(
                image=image, target_size=self.target_tensor_size, to_tensor=True
            ),
            cv2.resize(
                np.load(segment_path),
                (self.target_tensor_size, self.target_tensor_size),
                interpolation=cv2.INTER_NEAREST,
            ),
            # self.annotation[
            #     self.annotation[CONST.FILE_NAME].str.contains(
            #         file_name, case=False, na=False
            #     )
            # ].iloc[0].to_dict(),
        )

    def __len__(self):
        return len(self.data)
