import logging
import torch
import os
from pathlib import Path


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = os.cpu_count()
IMG_SIZE = (28, 28)
BATCH_SIZE = 32

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw"
MODEL_PATH = PROJECT_ROOT / "models"
# RESULTS_PATH = PROJECT_ROOT / "results"

DATA_PATH.mkdir(parents=True, exist_ok=True)
# RESULTS_PATH.mkdir(parents=True, exist_ok=True)
MODEL_PATH.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        # logging.FileHandler(RESULTS_PATH / "logs.log", mode='a')
    ]
)