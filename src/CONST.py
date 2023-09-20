# Paths
ANNOTATION_PATH = "../data/annotations/merged_wound_details_july_2022.csv"
ANNOTATION_PROCESSED_PATH = "../data/processed/annotations/wound_details.csv"
UNPROCESSED_IMAGES_DIR = "../data/images/"
PROCESSED_IMAGES_DIR = "../data/processed/roi/"
PROCESSED_SEGMENT_DIR = "../data/processed/segment/"

# Annotations
FILE_NAME = "ImageDetail.ImageFilename"
WOUND_RULER = "ImageDetail.hasRuler"
WOUND_TYPE = "ImageDetail.WoundType"
WOUND_BED = "ImageDetail.OtherWoundBedCategory"
WOUND_DEPTH = "WoundDetail.WoundDepth"
WOUND_LOCATION = "WoundDetail.Location"

# Region of interest
ROI_X = "ImageDetail.AreaOfInterestX"
ROI_X_WIDTH = "ImageDetail.AreaOfInterestWidth"
ROI_Y = "ImageDetail.AreaOfInterestY"
ROI_Y_HEIGHT = "ImageDetail.AreaOfInterestHeight"

# Filter columns
SELECT_COLUMN = [
    FILE_NAME,
    WOUND_RULER,
    WOUND_TYPE,
    WOUND_BED,
    WOUND_DEPTH,
    WOUND_LOCATION,
    ROI_X,
    ROI_X_WIDTH,
    ROI_Y,
    ROI_Y_HEIGHT,
]

# Segmentation model
SEG_MODEl_PATH = "../data/model/SegmentationLevel1.onnx"

# Segmentation input size
SEG_INPUT_SIZE = 336

# Segmentation annotation dictionary
SEGMENT_DICT = {
    0: "peri_wound",
    1: "wound",
    2: "skin",
    3: "background"
}
