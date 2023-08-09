import albumentations as A
import cv2
import numpy as np
import torch

import constraints.SEG_CONST as CONSTRAINTS
from entities.Image import Image
from utils.data_utils import init_samples, get_metadata
from utils.model_utils import load_onnx, init_session_onnx


if __name__ == "__main__":
    # Initialise the model
    model = load_onnx(CONSTRAINTS.MODEL_PATH)
    session = init_session_onnx(model)

    # Get the metadata
    metadata = get_metadata(
        CONSTRAINTS.METADATA_PATH,
        CONSTRAINTS.SAMPLE_COUNT,
        ("ImageDetail.hasRuler", None, False),
        ("ImageDetail.Status", "=", "ReadyForReview")
    )

    #

    print("Testing...")
