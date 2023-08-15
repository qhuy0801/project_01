import os
from datetime import datetime

from PIL import Image

import constraints.SEG_CONST as CONSTRAINTS
from services.segmentation_service import get_segmentation, resize_segmentation
from utils.data_utils import init_samples, get_metadata
from utils.io_utils import make_segmentation_dir, make_dir
from utils.model_utils import load_onnx, init_session_onnx

if __name__ == "__main__":
    # Initialise the model
    model = load_onnx(CONSTRAINTS.MODEL_PATH)
    session = init_session_onnx(model)

    # Get the metadata
    metadata = get_metadata(CONSTRAINTS.METADATA_PATH, CONSTRAINTS.SAMPLE_COUNT)

    # Create directory to save segmented image
    now = datetime.now()
    dt_string = now.strftime("%m%d%H%M")
    segmentation_dir = CONSTRAINTS.SEGMENTED_DIR + dt_string + "all"
    make_segmentation_dir(segmentation_dir)

    # # Init the samples
    samples = init_samples(
        metadata, CONSTRAINTS.FILENAME_COLUMN, CONSTRAINTS.IMAGES_DIR
    )

    # # Loop to push sample through model
    for sample in samples:
        sample.load_image()
        sample.get_model_input()
        fitted_segmentation = resize_segmentation(
            get_segmentation(sample, session), sample.image_matrix.shape[:-1][::-1]
        )
        sample.divide_segmentation(fitted_segmentation, CONSTRAINTS.SEGMENT_DICT)

        # Save file
        roi_folder = os.path.join(
            segmentation_dir, "roi", str(sample.metadata.get("ImageDetail.WoundType"))
        )
        make_dir(roi_folder)
        roi = sample.segments["peri_wound"] + sample.segments["wound"]
        Image.fromarray(roi).save(os.path.join(roi_folder, str(sample.file_name)))

    print("Done saving!")
