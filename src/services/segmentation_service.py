import cv2
import numpy as np


def get_segmentation(_sample, _session):
    """
    Get segmentation by pushing the sample through inference session
    :param _sample:
    :param _session:
    :return:
    """
    return np.argmax(_session.run(None, {"input": _sample.model_matrix})[0], axis=1)[0]


def resize_segmentation(_segmentation_matrix, _size):
    """
    Resize the segmentation matrix to fit the images
    :param _segmentation_matrix:
    :param _size:
    :return:
    """
    return cv2.resize(_segmentation_matrix, _size, interpolation=cv2.INTER_NEAREST)
