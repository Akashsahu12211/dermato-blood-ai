import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.model import DermatoBloodNetV2


BLOOD_GROUPS = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class FingerprintDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, abo_label, rh_label = self.samples[index]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read fingerprint image: {image_path}")
        image = cv2.createCLAHE(
            clipLimit=3.5, tileGridSize=(8, 8)
        ).apply(image)
        image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)
        return tensor, torch.tensor(abo_label), torch.tensor(rh_label)


def collect_samples(dataset_dir):
    samples = []
    for blood_group in BLOOD_GROUPS:
        folder = os.path.join(dataset_dir, blood_group)
        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"Missing blood-group folder '{folder}'. Expected folders: {BLOOD_GROUPS}"
            )
        abo_label = {"A": 0, "B": 1, "AB": 2, "O": 3}[blood_group.rstrip("+-")]
        # Keep labels aligned with RH_CLASSES in app/model.py: ["+", "-"].
        rh_label = 0 if blood_group.endswith("+") else 1
        files = [
            os.path.join(folder, name)
            for name in os.listdir(folder)
            if name.lower().endswith(IMAGE_EXTENSIONS)
        ]
        if not files:
            raise ValueError(f"No fingerprint images found in '{folder}'")
        samples.extend((path, abo_label, rh_label) for path in files)
        print(f"{blood_group}: {len(files)} images")
    return samples


def train_and_save_model():
    print("=" * 60)
    print("TRAINING ON LABELED FINGERPRINT FOLDERS")
    print("=" * 60)

    random.seed(42)
    torch.manual_seed(42)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_dir = os.path.join(base_dir, "data", "dataset_blood_group")
    samples = collect_samples(dataset_dir)
    random.shuffle(samples)

    validation_size = max(1, int(len(samples) * 0.2))
    training_size = len(samples) - validation_size
    train_set, validation_set = random_split(
        FingerprintDataset(samples),
        [training_size, validation_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
    validation_loader = DataLoader(
        validation_set, batch_size=32, shuffle=False, num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DermatoBloodNetV2().to(device)
    criterion_abo = nn.CrossEntropyLoss()
    criterion_rh = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_validation_score = -1.0
    best_state = None
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for images, abo_labels, rh_labels in train_loader:
            images = images.to(device)
            abo_labels = abo_labels.to(device)
            rh_labels = rh_labels.to(device)
            optimizer.zero_grad()
            abo_output, rh_output = model(images)
            loss = criterion_abo(abo_output, abo_labels) + criterion_rh(
                rh_output, rh_labels
            )
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        abo_correct = rh_correct = total = 0
        with torch.no_grad():
            for images, abo_labels, rh_labels in validation_loader:
                abo_output, rh_output = model(images.to(device))
                abo_correct += (abo_output.argmax(1).cpu() == abo_labels).sum().item()
                rh_correct += (rh_output.argmax(1).cpu() == rh_labels).sum().item()
                total += len(abo_labels)
        validation_score = (abo_correct + rh_correct) / (2 * total)
        print(
            f"Epoch [{epoch + 1}/10] "
            f"Loss: {running_loss / len(train_loader):.4f} | "
            f"Validation ABO: {abo_correct / total:.2%} | "
            f"Validation Rh: {rh_correct / total:.2%}"
        )
        if validation_score > best_validation_score:
            best_validation_score = validation_score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    model_path = os.path.join(base_dir, "models", "blood_group_model.pth")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(best_state, model_path)
    print(f"Trained model successfully saved to '{model_path}'.")


if __name__ == "__main__":
    train_and_save_model()
