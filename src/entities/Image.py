import cv2
import pandas as pd

from entities.File import File


class Image(File):

    img: cv2.Mat
    metadata: pd.Series

    def __init__(self, _file_name: str = "", _file_path: str = "") -> None:
        super().__init__(_file_name, _file_path)

    def load_img(self) -> None:
        self.img = cv2.cvtColor(cv2.imread(self.file_path), cv2.COLOR_BGR2RGB)

    def load_metadata(self, _metadata: pd.Series) -> None:
        self.metadata = _metadata
