import streamlit as st
from PIL import Image
import numpy as np
from kmeans_segmentation import segment_kmeans

st.title("🧠 K-Means Image Segmentation")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
k_value = st.slider("Choose number of clusters (K):", 2, 10, 4)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    with st.spinner("Segmenting using K-Means..."):
        segmented_img = segment_kmeans(np.array(image), k_value)

    st.image(segmented_img, caption=f"Segmented Image (K={k_value})", use_column_width=True)
