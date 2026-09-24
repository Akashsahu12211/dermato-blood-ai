import base64

import cv2
import numpy as np


def preprocess_fingerprint(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(
            "Selected image file is corrupt or invalid. Please upload a clear fingerprint image."
        )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(
        clipLimit=3.5, tileGridSize=(8, 8)
    ).apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    binarized = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )

    sift = cv2.SIFT_create(nfeatures=50)
    keypoints, descriptors = sift.detectAndCompute(enhanced, None)
    sift_img = cv2.drawKeypoints(
        enhanced,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        color=(0, 200, 255),
    )
    keypoint_count = len(keypoints) if keypoints is not None else 0
    ridge_density = float(np.mean(binarized == 255))
    ridge_variance = float(cv2.Laplacian(enhanced, cv2.CV_64F).var())

    resized = cv2.resize(enhanced, (128, 128))
    tensor_img = np.expand_dims(resized.astype(np.float32) / 255.0, axis=(0, 1))

    def to_b64(mat):
        _, buf = cv2.imencode(".png", mat)
        return "data:image/png;base64," + base64.b64encode(buf).decode("utf-8")

    vis_dict = {
        "raw": to_b64(img),
        "enhanced": to_b64(enhanced),
        "skeleton": to_b64(binarized),
        "sift": to_b64(sift_img),
        "ridge_density": round(ridge_density, 4),
        "ridge_variance": round(ridge_variance, 2),
        "sift_count": keypoint_count,
    }
    return tensor_img, vis_dict, enhanced, descriptors
