import streamlit as st
import requests
import os
from PIL import Image
import torch
import cv2
import time
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Set page config as the first command
st.set_page_config(page_title="Sign Language Translator", layout="wide")

# Load the ASL alphabet model from Hugging Face
@st.cache_resource
def load_asl_model():
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
    return processor, model

processor, model = load_asl_model()

# Function for ASL classification
def classify_asl(image):
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # ASL alphabet labels
    return labels[prediction % len(labels)]

# Streamlit UI
def main():
    st.title("Sign Language Translator")

    # Sidebar
    with st.sidebar:
        st.header("Menu")
        col1 = st.columns(1)[0]
        with col1:
            if st.button("📖 About Us", use_container_width=True):
                st.write(
                    "Welcome to the Sign Language Translator! Our platform uses advanced AI to help translate American Sign Language (ASL) alphabet gestures in real-time. "
                    "This project aims to bridge communication gaps and make sign language more accessible for everyone."
                )

            if st.button("📞 Contact Us", use_container_width=True):
                st.write(
                    "**Phone:** +1 (555) 987-6543  "
                    "**LinkedIn:** [Sign Language Translator](https://linkedin.com)  "
                    "**Facebook:** [Sign Language Translator](https://facebook.com)  "
                    "**Instagram:** [@signlanguage.ai](https://instagram.com)  "
                    "**Email:** contact@signlanguage.ai"
                )

            if st.button("💬 Feedback", use_container_width=True):
                st.text_area("We value your feedback! Please share your thoughts below:")

    tab1, tab2, tab3 = st.tabs(["Image Load", "Take Picture", "Live ASL"])

    with tab1:
        st.subheader("📸 Image Load")
        uploaded_image = st.file_uploader("Upload an image of an ASL alphabet gesture", type=["png", "jpg", "jpeg"])
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            gesture = classify_asl(image)
            st.success(f"Detected Gesture: {gesture}")

    with tab2:
        st.subheader("📷 Take Picture")
        camera_image = st.camera_input("Take a picture")
        if camera_image:
            image = Image.open(camera_image)
            st.image(image, caption="Captured Image", use_container_width=True)
            gesture = classify_asl(image)
            st.success(f"Detected Gesture: {gesture}")

    with tab3:
        st.subheader("📹 Live ASL")
        if st.button("Enable Cam"):
            cap = cv2.VideoCapture(0)
            stframe = st.image([])

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                gesture = classify_asl(image)
                frame = cv2.putText(frame, f"Gesture: {gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                stframe.image(frame, channels="BGR", use_container_width=True)
                time.sleep(1)
            cap.release()

# Ensure proper indentation
if __name__ == "__main__":
    main()