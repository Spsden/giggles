import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"

# Model settings
MODEL_PATH = MODELS_DIR / "mobilenetv3_emotion_small.pth"
MODEL2_PATH = MODELS_DIR / "FER_trained_model.pt"
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "happy", "surprise"]
EMOTION_DICT = {0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness',4: 'anger', 5: 'disgust', 6: 'fear'}

# Camera settings
CAMERA_ID = 0
DETECTION_INTERVAL = 1.0  # seconds