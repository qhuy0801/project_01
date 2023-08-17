import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.data_utils import get_metadata
from utils.io_utils import find_path

import constraints.SEG_CONST as SEG_CONST


CLASS_DICT = {

}

class WoundDataset(Dataset):
    def __init__(self, _wound_img_dir) -> None:
        super().__init__()
        self.data = []
        self.metadata = get_metadata(SEG_CONST.METADATA_PATH, SEG_CONST.SAMPLE_COUNT)
        self.transform = A.Compose([
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            A.Resize(512, 512, interpolation=cv2.INTER_NEAREST),
            ToTensorV2(),
        ])
        for _, row in self.metadata.iterrows():
            wound_img_path = find_path(_wound_img_dir, row[SEG_CONST.FILENAME_COLUMN])
            if wound_img_path is not None:
                self.data.append([wound_img_path, row[SEG_CONST.WOUND_LABEL_COLUMN]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> T_co:
        wound_img_path, label = self.data[index]
        wound_img = cv2.cvtColor(cv2.imread(wound_img_path), cv2.COLOR_BGR2RGB)
        return self.transform(image=wound_img)["image"], torch.tensor(label)
