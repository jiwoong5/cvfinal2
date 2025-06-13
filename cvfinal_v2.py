import os, time
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torchvision import transforms

import cv2
import numpy as np
import matplotlib.pyplot as plt

class StereoDepthKITTI(Dataset):
  def __init__(self, root_dir, transform=None):
    super().__init__()
    self.left_paths  = sorted(glob(os.path.join(root_dir, 'training/image_2/*_10.png')))
    self.right_paths = sorted(glob(os.path.join(root_dir, 'training/image_3/*_10.png')))
    self.disp_paths  = sorted(glob(os.path.join(root_dir, 'training/disp_occ_0/*.png')))

    print(f"left_images : {len(self.left_paths)}")
    print(f"right_images: {len(self.right_paths)}")
    print(f"disparity   : {len(self.disp_paths)}")

    self.transform = transform

  def __len__(self):
    return len(self.left_paths)

  def __getitem__(self, idx):
    img0 = Image.open(self.left_paths[idx]).convert('RGB')
    img1 = Image.open(self.right_paths[idx]).convert('RGB')
    disp = Image.open(self.disp_paths[idx])

    if self.transform:
      img0 = self.transform(img0)
      img1 = self.transform(img1)

      disp = transforms.Resize((224,224))(disp)
      disp = transforms.ToTensor()(disp)
      disp = disp / 1242.0

    return img0, img1, disp

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.down = nn.Identity()

    def forward(self, x):
        identity = self.down(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

# 원본 Feature Extractor (Residual Block 사용하지 않음)
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # Initial conv
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Simple conv layers without residual connections
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)

# 개선된 Feature Extractor (Residual Block 4개 사용)
class ImprovedFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        # Residual Block 4개 사용
        self.layers = nn.Sequential(
            ResidualBlock(32, 32, stride=1),
            ResidualBlock(32, 32, stride=1),
            ResidualBlock(32, 32, stride=1),
            ResidualBlock(32, 32, stride=1),
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.layers(x)
        return x

class CostVolume2DAggregation(nn.Module):
  def __init__(self, max_disp=192):
    super().__init__()
    self.max_disp = max_disp
    self.conv = nn.Sequential(
      nn.Conv2d(max_disp,64,3,1,1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
      nn.Conv2d(64,32,3,1,1),      nn.BatchNorm2d(32), nn.ReLU(inplace=True),
      nn.ConvTranspose2d(32,1,4,2,1)
    )
  def forward(self, lf, rf):
    B,C,H,W = lf.size()
    cost = lf.new_zeros(B, self.max_disp, H, W)
    for d in range(self.max_disp):
      if d>0:
        diff = F.l1_loss(lf[:,:,:,d:], rf[:,:,:,:-d], reduction='none')
        cost[:,d,:,d:] = diff.mean(1)
      else:
        diff = F.l1_loss(lf, rf, reduction='none')
        cost[:,d,:,:] = diff.mean(1)
    return self.conv(cost)

# 원본 StereoNet
class StereoNet(nn.Module):
  def __init__(self, max_disp=192):
    super().__init__()
    self.feat  = FeatureExtractor()
    self.agg2d = CostVolume2DAggregation(max_disp)
  def forward(self, l, r):
    return self.agg2d(self.feat(l), self.feat(r))

# 수정된 StereoNet
class ImprovedStereoNet(nn.Module):
  def __init__(self, max_disp=192):
    super().__init__()
    self.feat  = ImprovedFeatureExtractor()  # 수정된 Feature Extractor 사용
    self.agg2d = CostVolume2DAggregation(max_disp)
  def forward(self, l, r):
    return self.agg2d(self.feat(l), self.feat(r))

def calculate_rmse(model, dataloader, device):
    model.eval()
    model.to(device)
    total_squared_error = 0.0
    total_pixels = 0
    
    with torch.no_grad():
        for left_img, right_img, gt_disp in dataloader:
            left_img = left_img.to(device)
            right_img = right_img.to(device)
            gt_disp = gt_disp.to(device)
            
            pred_disp = model(left_img, right_img)
            pred_disp = pred_disp.squeeze(1)
            gt_disp = gt_disp.squeeze(1)

            mask = gt_disp > 0
            squared_error = ((pred_disp[mask] - gt_disp[mask]) ** 2).sum()
            total_squared_error += squared_error.item()
            total_pixels += mask.sum().item()

    rmse = (total_squared_error / total_pixels) ** 0.5
    return rmse

def train_model(model, dataloader, device, num_epochs, model_name):
    """모델 훈련 함수"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    model.to(device)
    best_loss = float('inf')
    model_dir = './models'
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, f'best_{model_name}_model.pth')
    
    print(f"\n=== Training {model_name} Model ===")
    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss = 0
        t0 = time.time()

        for l_img, r_img, gt_disp in dataloader:
            l_img, r_img, gt_disp = l_img.to(device), r_img.to(device), gt_disp.to(device)

            optimizer.zero_grad()
            out = model(l_img, r_img)
            pred = out.squeeze(1)
            gt = gt_disp.squeeze(1)

            mask = gt > 0
            loss = criterion(pred[mask], gt[mask])

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)

        if epoch % 10 == 0:
            elapsed = time.time() - t0
            print(f"Epoch {epoch}/{num_epochs} Loss: {avg_loss:.4f} Time: {elapsed:.1f}s")
    
    return save_path

def compare_models():
    """모델 비교 실험 메인 함수"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 데이터 준비
    root_dir = '../data_scene_flow'
    batch_size = 8
    num_workers = 2
    num_epochs = 20  # 비교 실험용으로 줄임
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    dataset = StereoDepthKITTI(root_dir=root_dir, transform=transform)
    
    # 데이터를 train/validation으로 분할
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, persistent_workers=True, pin_memory=True)
    
    print(f"Total samples: {len(dataset)}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # 원본 모델 훈련
    original_model = StereoNet(192)
    original_path = train_model(original_model, train_loader, device, num_epochs, "original")
    
    # 수정된 모델 훈련
    improved_model = ImprovedStereoNet(192)
    improved_path = train_model(improved_model, train_loader, device, num_epochs, "improved")
    
    # 모델 평가 및 비교
    print("\n=== Model Evaluation ===")
    
    # 원본 모델 평가
    original_model.load_state_dict(torch.load(original_path, map_location=device))
    original_rmse = calculate_rmse(original_model, val_loader, device)
    print(f"Original Model RMSE: {original_rmse:.4f}")
    
    # 수정된 모델 평가
    improved_model.load_state_dict(torch.load(improved_path, map_location=device))
    improved_rmse = calculate_rmse(improved_model, val_loader, device)
    print(f"Improved Model RMSE: {improved_rmse:.4f}")
    
    # 결과 비교
    improvement = ((original_rmse - improved_rmse) / original_rmse) * 100
    print(f"\nRMSE Improvement: {improvement:.2f}%")
    
    if improved_rmse < original_rmse:
        print("✅ Improved model (with Residual Blocks) performs better!")
    else:
        print("❌ Original model (simple convolutions) performs better.")
    
    # 모델 파라미터 수 비교
    original_params = sum(p.numel() for p in original_model.parameters())
    improved_params = sum(p.numel() for p in improved_model.parameters())
    print(f"\nModel Parameters:")
    print(f"Original (Simple Conv): {original_params:,}")
    print(f"Improved (ResNet): {improved_params:,}")
    print(f"Parameter increase: {((improved_params - original_params) / original_params) * 100:.1f}%")
    
    print(f"\n=== Experiment Summary ===")
    print(f"Original Model: Simple convolutional layers without skip connections")
    print(f"Improved Model: ResNet-style with 4 Residual Blocks")
    print(f"Key difference: Addition of residual connections for better gradient flow")

if __name__ == "__main__":
    compare_models()