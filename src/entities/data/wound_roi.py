import glob

import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import albumentations as A
from albumentations.pytorch import ToTensorV2


class WoundROI(Dataset):
    """
    Wound ROI dataset for training of the pattern
    """

    def __init__(
        self, dataset_dir: str = "../data/segmented/08151709all/roi/", image_size: int = 128
    ) -> None:
        super().__init__()

        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.data = []
        self.class_tuple = ()

        file_list = glob.glob(self.dataset_dir + "*")
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            self.class_tuple = (*self.class_tuple, class_name)
            for img_path in glob.glob(class_path + "/*.jpeg"):
                self.data.append([img_path, class_name])

        self.class_dict = dict(zip(self.class_tuple, range(self.class_tuple.__len__())))

        self.transform = A.Compose(
            [
                A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                A.Resize(self.image_size, self.image_size, interpolation=cv2.INTER_NEAREST),
                ToTensorV2(),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> T_co:
        image_path, label = self.data[index]
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        return self.transform(image=image)["image"], torch.tensor(
            self.class_dict[label]
        )
