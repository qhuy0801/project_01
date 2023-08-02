import cv2
import pandas as pd


class File:

    file_path: str
    file_name: str

    def __init__(
        self, _file_name: str = "", _file_path: str = ""
    ) -> None:
        super().__init__()
        self.file_name = _file_name
        self.file_path = _file_path
