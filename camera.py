import cv2
import numpy as np
from keras.models import load_model

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.model = load_model('model/emotion_model.h5', compile=False)
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (64, 64))
            roi = roi_gray.astype('float32') / 255.0
            roi = np.expand_dims(roi, axis=0)      # Shape: (1, 64, 64)
            roi = np.expand_dims(roi, axis=-1)     # Shape: (1, 64, 64, 1)

            preds = self.model.predict(roi, verbose=0)
            label = self.emotions[np.argmax(preds)]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
