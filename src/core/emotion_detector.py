import torch
import cv2
import numpy as np
from pathlib import Path
from torchvision.models import mobilenet_v3_small,mobilenet_v3_large
from torchvision import transforms
import torch.nn as nn
from src.config import MODEL_PATH, EMOTIONS
from PIL import Image


class EmotionDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.emotions = EMOTIONS
        # print(self.device)

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def load_model(self):
        try:
            self.model = mobilenet_v3_small(pretrained=False)
            self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, 7)
            self.model.load_state_dict(torch.load(MODEL_PATH,map_location=torch.device('cpu')))
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False




    def predict_emotion(self, face_image):
        if self.model is None:
            return None, 0.0

        # TODO: Add your preprocessing here
        # This depends on your model's requirements

        # Placeholder - replace with your model's preprocessing
        # processed_image = self.preprocess(face_image)
        # prediction = self.model(processed_image)
        # emotion_idx = prediction.argmax().item()
        # confidence = prediction.max().item()

        # For now, return dummy data
        return "happy", 0.8


if __name__ == "__main__":
    detector = EmotionDetector()
    if detector.load_model():
        print("Model test passed!")
    else:
        print("Model test failed!")