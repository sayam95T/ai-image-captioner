# app.py

import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load BLIP model and processor (cached to save memory)
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

st.set_page_config(page_title="AI Image Caption Generator", layout="centered")

# App title
st.title("🖼️ AI Image Caption Generator")
st.write("Upload an image and let AI describe it!")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Button to trigger caption generation
    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            processor, model = load_model()

            inputs = processor(images=image, return_tensors="pt")
            out = model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(out[0], skip_special_tokens=True)

        st.success("📝 Caption:")
        st.markdown(f"### {caption}")
