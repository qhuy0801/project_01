# Paths
ANNOTATION_PATH = "../data/annotations/merged_wound_details_july_2022.csv"
ANNOTATION_PROCESSED_PATH = "../resources/processed/annotations/wound_details.csv"
UNPROCESSED_IMAGES_DIR = "../data/images/"
PROCESSED_IMAGES_DIR = "../resources/processed/roi/"
PROCESSED_SEGMENT_DIR = "../resources/processed/segment/"
PROCESSED_EMBEDDING_DIR = "../resources/processed/embeddings/"

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

# Models output directory
OUTPUT_DIR = "../resources/output/"


# Model settings
class VAE_SETTING:
    """
    Setting for VAE_v1
    """
    # Identifier
    RUN_NAME = "vae_v3"

    # Initialisation
    INPUT_SIZE: int = 256
    DIM_CONFIG: [int] = [3, 4, 4, 4]
    LATENT_DIM: int = 512

    # Training
    BATCH_SIZE: int = 512
    CHECKPOINT_PATH: str = "../wound-data/output/vae_v3/09261520/vae_v3.pt"
    NUM_WORKERS: int = 1
    NUM_SAMPLES: int = 1
    EPOCHS = 10000
    MAX_LR = 5e-6
    DECAY_RATE = 1.


class VAE_SETTING_v3:
    """
    Setting for VAE_v3
    """
    # Identifier
    RUN_NAME = "vae_v3_1"

    # Initialisation
    INPUT_SIZE: int = 128
    ADDITIONAL_INPUT_SIZE: int = 256
    ENCODER_DIM: [int] = [3, 4]
    DECODER_DIM: [int] = [4, 4, 3]
    LATENT_DIM: int = 2048

    # Training
    BATCH_SIZE: int = 64
    CHECKPOINT_PATH: str = None
    NUM_WORKERS: int = 8
    NUM_SAMPLES: int = 1
    EPOCHS = 10000
    MAX_LR = 1e-4
    DECAY_RATE = 1.


class AE_SETTING_v1:
    """
    Setting for Autoencoder_v1
    """
    # Identifier
    RUN_NAME = "ae_v1"
    OUTPUT_DIR = "../resources/output/"

    # Initialisation
    INPUT_SIZE: int = 128

    # Training
    BATCH_SIZE: int = 64
    CHECKPOINT_PATH: str = None
    NUM_WORKERS: int = 8
    NUM_SAMPLES: int = 1
    EPOCHS = 5000
    MAX_LR: int = 5e-4
    LR_DECAY: int = 0.98
    MIN_LR: float = 1e-7
    LR_THRESHOLD: float = 0.3
    PATIENCE_LR: int = 15
    DECAY_RATE: float = .98


class VAE_SETTING_v4:
    """
    Setting for VAE_v4
    """
    # Identifier
    RUN_NAME = "vae_v4"
    OUTPUT_DIR = "../resources/output/"

    # Initialisation
    INPUT_SIZE: int = 128

    # Training
    BATCH_SIZE: int = 64
    CHECKPOINT_PATH: str = None
    NUM_WORKERS: int = 8
    NUM_SAMPLES: int = 1
    EPOCHS: int = 5000
    MAX_LR: float = 5e-4
    MIN_LR: float = 1e-7
    LR_THRESHOLD: float = 0.3
    PATIENCE_LR: int = 15
    DECAY_RATE: float = .98

class MULTI_HEADED_AE_SETTING:
    """
    Setting for Multi headed AE
    """
    # Identifier
    RUN_NAME = "multi_headed_ae"
    OUTPUT_DIR = "../resources/output/"

    # Initialisation
    INPUT_SIZE: int = 128

    # Simple AE checkpoint
    AE_CHECKPOINT = "../resources/checkpoint/ae_v1_0148.pt"

    # Training
    BATCH_SIZE: int = 64
    NUM_WORKERS: int = 8
    NUM_SAMPLES: int = 1
    EPOCHS: int = 5000
    MAX_LR: float = 5e-4
    MIN_LR: float = 1e-7
    LR_THRESHOLD: float = 0.3
    PATIENCE_LR: int = 15
    DECAY_RATE: float = .98

    # Additional decoder
    ADDITIONAL_MAX_LR: float = 1e-5 # previous 1e-4
    ADDITIONAL_MIN_LR: float = 1e-8 # previous 5e-8
    ADDITIONAL_DECAY_RATE: float = 0.95
    ADDITIONAL_LR_THRESHOLD: float = 0.1
    ADDITIONAL_LR_PATIENCE: int = 15
