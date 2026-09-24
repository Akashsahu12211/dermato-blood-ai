import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DermatoBloodNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.dropout = nn.Dropout(0.4)
        self.fc_shared = nn.Linear(256 * 8 * 8, 512)
        self.abo_head = nn.Linear(512, 4)
        self.rh_head = nn.Linear(512, 2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc_shared(x)))
        return self.abo_head(x), self.rh_head(x)


ABO_CLASSES = ["A", "B", "AB", "O"]
RH_CLASSES = ["+", "-"]


def classify_fingerprint_image(
    model, tensor_img, raw_enhanced_mat, ridge_density, sift_count, device="cpu"
):
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(tensor_img).to(device)
        abo_out, rh_out = model(x)
        raw_abo_probs = F.softmax(abo_out, dim=1).cpu().numpy()[0]
        raw_rh_probs = F.softmax(rh_out, dim=1).cpu().numpy()[0]

    abo_idx = int(np.argmax(raw_abo_probs))
    rh_idx = int(np.argmax(raw_rh_probs))

    return {
        "predicted_blood_group": f"{ABO_CLASSES[abo_idx]}{RH_CLASSES[rh_idx]}",
        "abo": ABO_CLASSES[abo_idx],
        "rh": RH_CLASSES[rh_idx],
        "abo_confidence": float(raw_abo_probs[abo_idx]),
        "rh_confidence": float(raw_rh_probs[rh_idx]),
        "abo_distribution": dict(zip(ABO_CLASSES, map(float, raw_abo_probs))),
        "rh_distribution": dict(zip(RH_CLASSES, map(float, raw_rh_probs))),
        "pattern_detected": "Fingerprint pattern analysis",
        "sift_count": sift_count,
    }
