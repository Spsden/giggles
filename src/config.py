import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"

# Model settings
MODEL_PATH = MODELS_DIR / "mobilenetv3_emotion_small.pth"
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Camera settings
CAMERA_ID = 0
DETECTION_INTERVAL = 1.0  # seconds