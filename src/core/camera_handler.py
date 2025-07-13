import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


class CameraHandler:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def start_camera(self):
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            print(f"[ERROR] Could not open camera with ID {self.camera_id}.")
            return False

        ret, _ = self.cap.read()
        if not ret:
            print(f"[ERROR] Failed to read from camera {self.camera_id}.")
            self.cap.release()
            self.cap = None
            return False

        print("[INFO] Camera started successfully.")
        return True

    def capture_frame(self):
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces

    def extract_and_preprocess_face(self, frame, bbox):
        x, y, w, h = bbox
        face_roi = frame[y:y + h, x:x + w]
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(face_rgb)
        tensor_image = self.transform(pil_image)
        return tensor_image.unsqueeze(0)

    def stop_camera(self):
        if self.cap:
            self.cap.release()