# IBM Bob Technology Utilization Document

## Project: DermatoBlood AI

**Author:** Akash Sahu  
**Institution:** Sharda University, Greater Noida, India

---

### Overview

This document details how IBM Bob technology was leveraged to architect, optimize, and deploy DermatoBlood AI—a non-invasive auxiliary biometric triage system for blood phenotype estimation.

### Key Workflows & AI Interventions Powered by IBM Bob

1. **Neural Network Architecture Synthesis (`app/model.py`):**
   - Engineered `DermatoBloodNet` using IBM Bob to fuse 128-d OpenCV SIFT keypoints with a 2048-d ResNet-50 bottleneck feature vector.
   - Implemented dual classification heads for multi-class ABO softmax estimation and binary Rh factor sigmoid prediction.

2. **Custom Loss Function Engineering (`app/loss.py`):**
   - Designed `MultiTaskFocalLoss` ($\gamma=2.0$, task weights $\alpha=0.7, \beta=0.3$) to penalize easy examples and resolve class imbalance for rare blood groups ($AB^-, B^-$).

3. **Explainable AI Engine (`app/gradcam.py`):**
   - Configured Gradient-Weighted Class Activation Mapping (Grad-CAM) targeting `layer4.2.conv3` to verify attention on central core ridge loops and delta points, resolving $A^+/O^+$ confusion.

4. **FastAPI Microservice Deployment (`app/main.py` & `run.py`):**
   - Structured asynchronous RESTful API endpoints for raw optical image decoding, feature normalization, and automated clinical report compilation.
