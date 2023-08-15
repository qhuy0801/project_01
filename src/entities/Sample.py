import albumentations as A
import cv2
import numpy as np

from entities.Image import Image


class Sample(Image):
    # Model input matrix
    model_matrix: np.ndarray

    # Model dimension order
    dimension_order = (0, 3, 1, 2)

    # Transformation template for sample manipulation
    transform_template = A.Compose(
        [
            A.Resize(336, 336, interpolation=cv2.INTER_NEAREST),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Segmented parts
    segments = {}

    def get_model_input(self):
        """
        Get a model-ready matrix
        :return:
        """
        self.model_matrix = np.expand_dims(
            self.transform_template(image=self.image_matrix)["image"], axis=0
        ).transpose(self.dimension_order)

    def divide_segmentation(self, _segment_matrix, _segment_dict):
        for key in _segment_dict:
            _mask = _segment_matrix == key
            self.segments[_segment_dict.get(key)] = cv2.bitwise_and(
                self.image_matrix, self.image_matrix, mask=_mask.astype(np.uint8)
            )
