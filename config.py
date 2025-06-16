import torch
import logging
from transformers import BitsAndBytesConfig

# ===== CURRENT RUN PARAMETERS ===== #
DATA_DIR = "emea_data/en-et"
SOURCE_LANG = "English"
TARGET_LANG = "Estonian"
DATASET_UTILIZATION = 0.033  # amount of entire dataset used
# ================================== #

CACHE_DIR = "./model_cache"
MODEL_NAME = "google/gemma-3-4b-pt"
OUTPUT_DIR = "./output"

# Logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Quantization config
NF4_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
