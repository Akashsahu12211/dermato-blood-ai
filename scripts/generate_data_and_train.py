import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.model import DermatoBloodNetV2

def generate_fingerprint_pattern(pattern_type, seed):
    np.random.seed(seed)
    img = np.zeros((128, 128), dtype=np.float32)
    x = np.linspace(-3, 3, 128)
    y = np.linspace(-3, 3, 128)
    X, Y = np.meshgrid(x, y)
    
    if pattern_type == 0:
        Z = np.sin(6 * (X**2 + Y**2)) + 0.25 * np.random.randn(128, 128)
    elif pattern_type == 1:
        R = np.sqrt(X**2 + Y**2)
        Z = np.cos(10 * R) + 0.25 * np.random.randn(128, 128)
    elif pattern_type == 2:
        Z = np.sin(5 * X + Y**2) + 0.25 * np.random.randn(128, 128)
    else:
        Z = np.sin(4 * X) * np.cos(4 * Y) + 0.25 * np.random.randn(128, 128)
        
    Z_norm = cv2.normalize(Z, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return Z_norm

def train_and_save_model():
    print("="*60)
    print("TRAINING DERMATOBLOODNET V2 MODEL ON SYNTHETIC DERMATOGLYPHIC SAMPLES")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DermatoBloodNetV2().to(device)
    
    criterion_abo = nn.CrossEntropyLoss()
    criterion_rh = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    X_list, Y_abo_list, Y_rh_list = [], [], []
    for i in range(640):
        pat_type = i % 4
        img = generate_fingerprint_pattern(pat_type, seed=i)
        norm_img = img.astype(np.float32) / 255.0
        X_list.append(np.expand_dims(norm_img, axis=0))
        
        abo_label = i % 4
        rh_label = (i // 4) % 2
        Y_abo_list.append(abo_label)
        Y_rh_list.append(rh_label)
        
    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32)
    Y_abo_tensor = torch.tensor(Y_abo_list, dtype=torch.long)
    Y_rh_tensor = torch.tensor(Y_rh_list, dtype=torch.long)
    
    dataset = torch.utils.data.TensorDataset(X_tensor, Y_abo_tensor, Y_rh_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    model.train()
    epochs = 10
    for epoch in range(epochs):
        running_loss = 0.0
        for b_x, b_abo, b_rh in loader:
            b_x, b_abo, b_rh = b_x.to(device), b_abo.to(device), b_rh.to(device)
            optimizer.zero_grad()
            out_abo, out_rh = model(b_x)
            loss_abo = criterion_abo(out_abo, b_abo)
            loss_rh = criterion_rh(out_rh, b_rh)
            loss = loss_abo + loss_rh
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss / len(loader):.4f}")
        
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "blood_group_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Trained model successfully saved to '{model_path}'.")

if __name__ == "__main__":
    train_and_save_model()
