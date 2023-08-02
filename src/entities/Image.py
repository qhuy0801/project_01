import cv2
import numpy as np
import pandas as pd

from entities.File import File


class Image(File):

    # Image pixel matrix in RGB
    img: np.ndarray

    # Other metadata
    metadata: pd.Series

    def __init__(self, _file_name: str = "", _file_path: str = "") -> None:
        """
        Constructor
        :param _file_name:
        :param _file_path:
        """
        super().__init__(_file_name, _file_path)

    def load_image(self) -> None:
        """
        Load image from file to RGB matrix using OpenCV
        :return:
        """
        self.img = cv2.cvtColor(cv2.imread(self.file_path), cv2.COLOR_BGR2RGB)

    def load_metadata(self, _metadata: pd.Series) -> None:
        """
        Load metadata based on row in .csv file
        :param _metadata:
        :return:
        """
        self.metadata = _metadata
