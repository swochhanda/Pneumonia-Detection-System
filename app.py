import os

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Pneumonia Detection System",
    layout="centered",
)

st.title("Pneumonia Detection System")
st.subheader("Powered by ResNet101V2")

MODEL_PATH = os.path.join("Saved model", "mymodel.keras")


@st.cache_resource
def load_my_model():
    return load_model(MODEL_PATH)


with st.spinner("Loading AI model..."):
    model = load_my_model()


uploaded_image = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png", "webp"],
)


def preprocess(img):
    img_resized = img.resize((224, 224))
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return img_resized, arr


if uploaded_image is not None:
    img = Image.open(uploaded_image).convert("RGB")
    original_width, original_height = img.size

    resized_img, processed_array = preprocess(img)

    col1, col2 = st.columns(2)

    with col1:
        st.write("Original Image")
        st.image(img)
        st.caption(f"Dimensions: {original_width} × {original_height}")

    with col2:
        st.write("Preprocessed Image")
        st.image(resized_img)
        st.caption("Dimensions: 224 × 224")

    with st.spinner("Analyzing X-ray image..."):
        prediction = model(processed_array, training=False).numpy()
        probability = prediction[0][0]

    st.divider()

    if probability >= 0.5:
        result = "PNEUMONIA"
        confidence = probability * 100
        st.error(f"Prediction: {result}")
    else:
        result = "NORMAL"
        confidence = (1 - probability) * 100
        st.success(f"Prediction: {result}")

    st.info(f"Confidence: {confidence:.2f}%")