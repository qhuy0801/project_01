import albumentations as A
import cv2
import numpy as np
import torch

import constraints.SEG_CONST as CONSTRAINTS
from entities.Image import Image
from utils.data_utils import init_images, get_metadata


if __name__ == "__main__":

    # Get the metadata
    metadata = get_metadata(
        CONSTRAINTS.METADATA_PATH,
        CONSTRAINTS.SAMPLE_COUNT,
        ("ImageDetail.hasRuler", None, True)
    )

    print("Testing...")
