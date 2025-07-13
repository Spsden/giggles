import time
import threading
from src.core.emotion_detector import EmotionDetector
from src.core.camera_handler import CameraHandler
from src.core.emotion_processor import EmotionProcessor
# from src.core.notification_manager import NotificationManager
from src.config import MODEL_PATH


class BackgroundService:
    def __init__(self):
        self.emotion_detector = EmotionDetector()
        self.camera_handler = CameraHandler()
        self.emotion_processor = EmotionProcessor()
        # self.notification_manager = NotificationManager()

        self.is_running = False
        self.detection_thread = None
        self.detection_interval = 1.0

    def start_detection(self):
        if self.is_running:
            return

        print("Trying to load model")
        self.emotion_detector.load_model()
        print("Trying to start camera")
        self.camera_handler.start_camera()

        self.is_running = True
        self.detection_thread = threading.Thread(target=self.detection_loop)
        self.detection_thread.start()

    def stop_detection(self):
        self.is_running = False
        if self.detection_thread:
            self.detection_thread.join()
        self.camera_handler.stop_camera()

    def detection_loop(self):
        while self.is_running:
            try:
                frame = self.camera_handler.capture_frame()
                if frame is None:
                    continue

                faces = self.camera_handler.detect_faces(frame)

                if len(faces) != 0:
                    # Find largest face by area
                    print("HIIII")
                    largest_face = max(faces, key=lambda bbox: bbox[2] * bbox[3])
                    face_tensor = self.camera_handler.extract_and_preprocess_face(frame, largest_face)
                    emotion, confidence = self.emotion_detector.predict_emotion(face_tensor)
                    print(emotion, confidence)


                    result = self.emotion_processor.process_emotion_result(emotion, confidence)
                    # if result and result['should_notify']:
                    #     self.notification_manager.send_emotion_notification(
                    #         result['emotion'],
                    #         result['confidence']
                    #     )

                time.sleep(self.detection_interval)

            except Exception as e:
                print(f"Error in detection loop: {e}")
                time.sleep(1)

if __name__ == "__main__":
    bg = BackgroundService()
    bg.start_detection()