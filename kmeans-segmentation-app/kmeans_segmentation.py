import numpy as np
import cv2

def segment_kmeans(image, k=4):
    image_resized = cv2.resize(image, (256, 256))  # Optional
    img_data = image_resized.reshape((-1, 3))
    img_data = np.float32(img_data)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(img_data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_img = segmented_data.reshape(image_resized.shape)

    return segmented_img
