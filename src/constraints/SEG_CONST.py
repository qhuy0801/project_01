from utils.config_utils import read_config

CONFIG_FILE = "../../config/segmentation-config.yml"
config = read_config(CONFIG_FILE)

# Settings for file paths
METADATA_PATH = config["data"]["metadata_path"]
IMAGES_DIR = config["data"]["images_dirs"]

# Segmentation output directory
SEGMENTED_DIR = config["data"]["segmentation_output"]

# Settings for .csv metadata file
FILENAME_COLUMN = config["data"]["column_names"]["file_name"]

# Setting for segmentation models
MODEL_PATH = config["model"]["path"]
MODEL_INPUT_HEIGHT = config["model"]["input_size"]["height"]
MODEL_INPUT_WIDTH = config["model"]["input_size"]["width"]

# Setting for samples
SAMPLE_COUNT = config["samples"]["sample_count"]


# Computational device
DEVICE = config["device"]

# Visualising setting
SEGMENT_CMAP = config["visualisations"]["segment_cmap"]

# Segmentation annotation dictionary
SEGMENT_DICT = {
    0: "peri_wound",
    1: "wound",
    2: "skin",
    3: "background"
}
