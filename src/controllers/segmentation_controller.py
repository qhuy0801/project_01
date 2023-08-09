import albumentations as A
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

import constraints.SEG_CONST as CONSTRAINTS
from entities.Image import Image
from utils.data_utils import init_samples, get_metadata
from utils.model_utils import load_onnx, init_session_onnx
from utils.visualise_utils import visualise_layers
from services.segmentation_service import get_segmentation, resize_segmentation


if __name__ == "__main__":
    # Initialise the model
    model = load_onnx(CONSTRAINTS.MODEL_PATH)
    session = init_session_onnx(model)

    # Get the metadata
    metadata = get_metadata(
        CONSTRAINTS.METADATA_PATH,
        CONSTRAINTS.SAMPLE_COUNT,
        ("ImageDetail.hasRuler", None, False),
        ("ImageDetail.Status", "=", "ReadyForReview"),
    )

    # Init the samples
    samples = init_samples(
        metadata, CONSTRAINTS.FILENAME_COLUMN, CONSTRAINTS.IMAGES_DIR
    )

    # Loop to push sample through model
    for sample in samples:
        sample.load_image()
        sample.get_model_input()
        fitted_segmentation = resize_segmentation(
            get_segmentation(sample, session), sample.image_matrix.shape[:-1][::-1]
        )
        visualise_layers(
            (sample.image_matrix, CONSTRAINTS.SEGMENT_CMAP, 1),
            (fitted_segmentation, CONSTRAINTS.SEGMENT_CMAP, 0.5)
        )

    print("Testing...")
