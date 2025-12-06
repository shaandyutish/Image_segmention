🖼️ Image Segmentation Using K-Means & DBSCAN
🚀 A Machine Learning Project for Segmentation & Region Extraction

This project implements image segmentation using two clustering algorithms:

K-Means Clustering (color-based segmentation)

DBSCAN (density-based segmentation, useful for noisy and complex images)

It includes a complete pipeline for:
✔ Preprocessing images
✔ Applying K-Means or DBSCAN
✔ Visualizing segmented outputs
✔ Running the solution as a Streamlit Web App
✔ Supporting both single-image and batch processing

📌 Features

🎯 K-Means Segmentation
* Clusters image pixels into K color groups
* Produces clean segmented output
* Adjustable number of clusters

🎯 DBSCAN Segmentation
* Detects arbitrary-shaped regions
* Excellent for noisy & irregular patterns
* Unsupervised parameters (eps, min_samples)

🧵 End-to-End Pipeline
* Load → Resize → Normalize → Cluster → Visualize

🖥 Streamlit Application
* Upload images
* Select algorithm
* Generate segmentation in real-time

📦 Easy Deployment
* Works on Streamlit Cloud
* Includes requirements.txt
* Compatible with Kaggle Notebook
