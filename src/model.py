from tensorflow.keras.models import load_model

def load_image_model(path="models/cnn_image_model.h5"):
    return load_model(path)
