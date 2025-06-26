import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("models/cnn_image_model.h5")

# Image size used during training
IMAGE_SIZE = (224, 224)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)[0][0]
    label = "Fake" if prediction > 0.5 else "Real"
    confidence = float(prediction) if prediction > 0.5 else 1 - float(prediction)
    return label, round(confidence * 100, 2)
