import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import CONST
from utils import find_path, load_image, forward_transform, load_onnx, init_session_onnx
from utils.data_utils import remove_image_extension


def get_segmentation(_sample, _session):
    """
    Get segmentation by pushing the sample through inference session
    :param _sample:
    :param _session:
    :return:
    """
    return np.argmax(_session.run(None, {"input": _sample})[0], axis=1)[0]


def resize_segmentation(_segmentation_matrix, _size):
    """
    Resize the segmentation matrix to fit the images
    :param _segmentation_matrix:
    :param _size:
    :return:
    """
    return cv2.resize(_segmentation_matrix, _size, interpolation=cv2.INTER_NEAREST)


if __name__ == "__main__":
    # Read the annotations
    df = pd.read_csv(CONST.ANNOTATION_PROCESSED_PATH)

    # Initialise the model
    model = load_onnx(CONST.SEG_MODEl_PATH)
    session = init_session_onnx(model)

    # For each record, load file
    for _, row in df.iterrows():
        # Get file path
        file_path = find_path(
            filename=row[CONST.FILE_NAME], root_dir=CONST.UNPROCESSED_IMAGES_DIR
        )

        # Skip the missing file
        if file_path is None:
            continue

        # Crop the ROI
        if row[CONST.ROI_Y_HEIGHT] != 0 and row[CONST.ROI_X_WIDTH] != 0:
            image = load_image(file_path)[
                row[CONST.ROI_Y] : (row[CONST.ROI_Y] + row[CONST.ROI_Y_HEIGHT]),
                row[CONST.ROI_X] : (row[CONST.ROI_X] + row[CONST.ROI_X_WIDTH]),
            ]
        else:
            image = load_image(file_path)

        # Prepare for tensor
        model_input = np.expand_dims(
            forward_transform(
                image=image, target_size=CONST.SEG_INPUT_SIZE, to_tensor=False
            ),
            axis=0,
        ).transpose((0, 3, 1, 2))

        # Get segments
        segment = get_segmentation(_sample=model_input, _session=session)
        segment = resize_segmentation(
            _segmentation_matrix=segment, _size=image.shape[:-1][::-1]
        )

        # Save images and segments
        Image.fromarray(image).save(
            os.path.join(CONST.PROCESSED_IMAGES_DIR, row[CONST.FILE_NAME])
        )
        np.save(
            os.path.join(
                CONST.PROCESSED_SEGMENT_DIR,
                remove_image_extension(row[CONST.FILE_NAME]),
            )
            + ".npy",
            arr=segment,
        )

    print("Done segmentation!")
