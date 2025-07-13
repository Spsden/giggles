import time
from collections import deque
from datetime import datetime


class EmotionProcessor:
    def __init__(self):
        self.emotion_history = deque(maxlen=10)  # Last 10 emotions
        self.last_notification_time = {}
        self.notification_cooldown = 30  # seconds
        self.confidence_threshold = 0.6

    def process_emotion_result(self, emotion, confidence):
        current_time = time.time()

        if confidence < self.confidence_threshold:
            return None

        self.emotion_history.append({
            'emotion': emotion,
            'confidence': confidence,
            'timestamp': current_time
        })

        if self.should_notify(emotion, current_time):
            return {
                'emotion': emotion,
                'confidence': confidence,
                'should_notify': True
            }

        return {
            'emotion': emotion,
            'confidence': confidence,
            'should_notify': False
        }

    def should_notify(self, emotion, current_time):
        if emotion in self.last_notification_time:
            time_since_last = current_time - self.last_notification_time[emotion]
            if time_since_last < self.notification_cooldown:
                return False

        if len(self.emotion_history) >= 3:
            recent_emotions = [e['emotion'] for e in list(self.emotion_history)[-3:]]
            if recent_emotions.count(emotion) >= 2:  # Consistent emotion
                self.last_notification_time[emotion] = current_time
                return True

        return False

    def get_emotion_history(self):
        return list(self.emotion_history)