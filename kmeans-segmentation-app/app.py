import streamlit as st
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import matplotlib.pyplot as plt
import io

# --- Clustering function ---
def cluster_eval(features, k=4, label_shape=None):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(features)
    dbi = davies_bouldin_score(features, labels)
    # Subsample for silhouette score
    sample_feats, sample_labels = resample(features, labels, n_samples=min(5000, len(features)), random_state=42)
    sil = silhouette_score(sample_feats, sample_labels)
    return labels.reshape(label_shape), dbi, sil

# --- Image processing function ---
def process_image_streamlit(image_np):
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
    image_lab = cv2.cvtColor(image_np, cv2.COLOR_BGR2Lab)
    h, w, _ = image_rgb.shape

    # Feature extraction
    R = image_rgb[:, :, 0].reshape(-1, 1).astype(np.float32)
    H = image_hsv[:, :, 0].reshape(-1, 1).astype(np.float32)
    L = image_lab[:, :, 0].reshape(-1, 1).astype(np.float32)
    rgb = image_rgb.reshape(-1, 3).astype(np.float32)

    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    xy = np.stack([x_coords, y_coords], axis=2).reshape((-1, 2)).astype(np.float32)

    scaler = StandardScaler()
    R_scaled = scaler.fit_transform(R)
    H_scaled = scaler.fit_transform(H)
    L_scaled = scaler.fit_transform(L)
    rgb_scaled = scaler.fit_transform(rgb)
    xy_scaled = scaler.fit_transform(xy)

    # RGB+XY optimization
    spatial_weights = [0.2, 0.5, 1.0]
    k_values = [3, 4]
    best_rgbxy = {'dbi': np.inf}

    for sw in spatial_weights:
        for k in k_values:
            combined = np.hstack([rgb_scaled, xy_scaled * sw])
            seg, dbi, sil = cluster_eval(combined, k, (h, w))
            if dbi < best_rgbxy['dbi']:
                best_rgbxy = {
                    'seg': seg,
                    'dbi': dbi,
                    'sil': sil,
                    'sw': sw,
                    'k': k
                }

    seg_R, dbi_R, sil_R = cluster_eval(R_scaled, k=4, label_shape=(h, w))
    seg_H, dbi_H, sil_H = cluster_eval(H_scaled, k=4, label_shape=(h, w))
    seg_L, dbi_L, sil_L = cluster_eval(L_scaled, k=4, label_shape=(h, w))

    # Plotting to memory (instead of file)
    fig = plt.figure(figsize=(15, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")

    plt.subplot(2, 3, 2)
    plt.imshow(seg_R, cmap='viridis')
    plt.title(f"R Channel\nDBI: {dbi_R:.3f}, Sil: {sil_R:.3f}")

    plt.subplot(2, 3, 3)
    plt.imshow(seg_H, cmap='viridis')
    plt.title(f"H Channel\nDBI: {dbi_H:.3f}, Sil: {sil_H:.3f}")

    plt.subplot(2, 3, 4)
    plt.imshow(seg_L, cmap='viridis')
    plt.title(f"L Channel\nDBI: {dbi_L:.3f}, Sil: {sil_L:.3f}")

    plt.subplot(2, 3, 5)
    plt.imshow(best_rgbxy['seg'], cmap='viridis')
    plt.title(f"RGB+XY (w={best_rgbxy['sw']}, k={best_rgbxy['k']})\nDBI: {best_rgbxy['dbi']:.3f}, Sil: {best_rgbxy['sil']:.3f}")

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    return buf

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🧠 K-Means Image Segmentation (Optimized)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)

    with st.spinner("Processing..."):
        result_buf = process_image_streamlit(img_bgr)

    st.image(result_buf, caption="Segmented Output", use_column_width=True)
