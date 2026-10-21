import os
from pathlib import Path
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Dataset & Hyperparameters
RANDOM_SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.4
EARLY_STOPPING_PATIENCE = 6

# Data Splits
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Augmentation Parameters
AUGMENTATION_CONFIG = {
    "rotation_degrees": 15,
    "brightness_jitter": 0.15,
    "contrast_jitter": 0.15,
    "crop_scale": (0.85, 1.0),
    "hflip_prob": 0.5,
}

# ImageNet Normalization Constants
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Device Configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# File Paths for Artifacts
SPLIT_FILE = DATA_DIR / "train_val_test_split.json"
CLASS_MAPPING_FILE = MODELS_DIR / "class_mapping.json"
MARKETPLACE_CATALOG_FILE = DATA_DIR / "marketplace_catalog.json"

# Model Configuration (Xception - 98.09% Test Accuracy)
MODEL_NAME = "xception"
MODEL_PATH = MODELS_DIR / "xception.pth"
XCEPTION_MODEL_PATH = MODEL_PATH
BEST_MODEL_PATH = MODEL_PATH
EVALUATION_METRICS_FILE = OUTPUTS_DIR / "evaluation_metrics.json"
CLASSIFICATION_REPORT_FILE = OUTPUTS_DIR / "classification_report.txt"

# Visualization Output Paths
ACCURACY_CURVE_PATH = OUTPUTS_DIR / "accuracy_curve.png"
LOSS_CURVE_PATH = OUTPUTS_DIR / "loss_curve.png"
CONFUSION_MATRIX_PATH = OUTPUTS_DIR / "confusion_matrix.png"
SAMPLE_PREDICTIONS_PATH = OUTPUTS_DIR / "sample_predictions.png"
CORRECT_PREDICTIONS_PATH = OUTPUTS_DIR / "correct_predictions.png"
INCORRECT_PREDICTIONS_PATH = OUTPUTS_DIR / "incorrect_predictions.png"
