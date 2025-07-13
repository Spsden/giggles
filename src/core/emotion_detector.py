import cv2
from torchvision.models import mobilenet_v3_small
from src.config import MODEL2_PATH, EMOTION_DICT
from src.core.test_model import  *


class EmotionDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.emotions = EMOTION_DICT
        # print(self.device)

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def load_model(self):
        try:
            self.model = mobilenet_v3_small(pretrained=False)
            self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, 7)
            self.model.load_state_dict(torch.load(MODEL2_PATH,map_location=torch.device('cpu')))
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False




    def predict_emotion(self, face_tensor):
        if self.model is None:
            return None, 0.0

        with torch.no_grad():
            outputs = self.model(face_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            emotion_idx = predicted.item()
            confidence_score = confidence.item()
            return self.emotions[emotion_idx], confidence_score
        # with torch.no_grad():
        #     models.test_model.eval()
        #     log_ps = model.cpu()(X)
        #     ps = torch.exp(log_ps)
        #     top_p, top_class = ps.topk(1, dim=1)
        #     pred = emotion_dict[int(top_class.numpy())]



# if __name__ == "__main__":
#     detector = EmotionDetector()
#     if detector.load_model():
#         print("Model test passed!")
#     else:
#         print("Model test failed!")