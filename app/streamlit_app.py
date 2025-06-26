import streamlit as st
from src.predict_image import predict_image
import os

st.title("🧠 Deepfake Detection")
st.markdown("Upload an image to check whether it's real or fake.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    label, confidence = predict_image(image_path)
    st.markdown(f"### Prediction: `{label}` ({confidence}% confidence)")
