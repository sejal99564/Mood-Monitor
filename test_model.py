import os
import cv2
import numpy as np
from keras.models import load_model

# 🌍 Define paths
base_dir = os.getcwd()
model_path = os.path.join(base_dir, 'model', 'emotion_model.h5')
image_path = os.path.join(base_dir, 'test_clean.jpg')

# 🏷️ Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 🖼️ Load and preprocess image
print(f"📂 Current Directory: {base_dir}")
print(f"📦 Model Path: {model_path}")
print(f"🖼️  Image Path: {image_path}")
print("")

if not os.path.exists(image_path):
    raise FileNotFoundError(f"🚫 Image file not found at {image_path}")

image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"🚫 Failed to load image using OpenCV: '{image_path}'")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray, (64, 64))  # ✅ Resize to expected input
normalized = resized / 255.0
reshaped = np.reshape(normalized, (1, 64, 64, 1))  # ✅ Add batch and channel dimension

# 📥 Load model (WITHOUT compiling)
model = load_model(model_path, compile=False)

# 🤖 Predict emotion
prediction = model.predict(reshaped)
emotion_index = np.argmax(prediction)
predicted_emotion = emotion_labels[emotion_index]

# 🎯 Show result
print(f"\n🧠 Predicted Emotion: {predicted_emotion}")
