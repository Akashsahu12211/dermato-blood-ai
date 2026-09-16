import cv2
import numpy as np
import base64

def preprocess_fingerprint(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Selected image file is corrupt or invalid. Please upload a clear fingerprint image.")

    # 1. Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization) - Paper Sec 3.1
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 3. Gaussian Blur Noise Reduction
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # 4. Adaptive Binarization (Ridge Skeleton)
    binarized = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # 5. SIFT Minutiae Feature Extraction (As per Elsevier 2025 Paper Sec 3.2)
    sift = cv2.SIFT_create(nfeatures=50)
    keypoints, descriptors = sift.detectAndCompute(enhanced, None)
    
    # Draw keypoints on image (Rich Keypoints showing scale & orientation)
    sift_img = cv2.drawKeypoints(
        enhanced, keypoints, None, 
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, 
        color=(0, 200, 255)
    )
    
    keypoint_count = len(keypoints) if keypoints is not None else 0

    # Calculate Biological Dermatoglyphic Feature Metrics
    ridge_pixels = np.sum(binarized == 255)
    total_pixels = binarized.size
    ridge_density = float(ridge_pixels / total_pixels)
    
    # Spatial Frequency & Ridge Variance (Micro-Topology Invariants)
    laplacian_var = float(cv2.Laplacian(enhanced, cv2.CV_64F).var())
    
    # Resize to model input shape (128x128)
    resized = cv2.resize(enhanced, (128, 128))
    norm_img = resized.astype(np.float32) / 255.0
    tensor_img = np.expand_dims(norm_img, axis=(0, 1)) # [1, 1, 128, 128]

    def to_b64(mat):
        _, buf = cv2.imencode('.png', mat)
        return "data:image/png;base64," + base64.b64encode(buf).decode('utf-8')

    vis_dict = {
        "raw": to_b64(img),
        "enhanced": to_b64(enhanced),
        "skeleton": to_b64(binarized),
        "sift": to_b64(sift_img),
        "ridge_density": float(round(ridge_density, 4)),
        "ridge_variance": float(round(laplacian_var, 2)),
        "sift_count": keypoint_count
    }
    
    return tensor_img, vis_dict, enhanced, descriptors
