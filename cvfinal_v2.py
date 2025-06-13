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

class StereoDepthKITTITest(Dataset):
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.left_paths  = sorted(glob(os.path.join(root_dir, 'testing/image_2/*_10.png')))
        self.right_paths = sorted(glob(os.path.join(root_dir, 'testing/image_3/*_10.png')))

        print(f"[TEST] Left images : {len(self.left_paths)}")
        print(f"[TEST] Right images: {len(self.right_paths)}")

        self.transform = transform

    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx):
        img0 = Image.open(self.left_paths[idx]).convert('RGB')
        img1 = Image.open(self.right_paths[idx]).convert('RGB')

        if self.transform:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1
    
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
    
    #print(f"Total samples: {len(dataset)}")
    #print(f"Train samples: {len(train_dataset)}")
    #print(f"Validation samples: {len(val_dataset)}")
    
    # 원본 모델 훈련
    original_model = StereoNet(192)
    #original_path = train_model(original_model, train_loader, device, num_epochs, "original")
    
    # 수정된 모델 훈련
    improved_model = ImprovedStereoNet(192)
    #improved_path = train_model(improved_model, train_loader, device, num_epochs, "improved")
    
    # 모델 평가 및 비교
    print("\n=== Model Evaluation ===")
    
    # 원본 모델 평가
    original_model.load_state_dict(torch.load("./models/best_original_model.pth", map_location=device))
    original_rmse = calculate_rmse(original_model, val_loader, device)
    print(f"Original Model RMSE: {original_rmse:.4f}")
    
    # 수정된 모델 평가
    improved_model.load_state_dict(torch.load("./models/best_improved_model.pth", map_location=device))
    improved_rmse = calculate_rmse(improved_model, val_loader, device)
    print(f"Improved Model RMSE: {improved_rmse:.4f}")
    
    improved_model.load_state_dict(torch.load("./models/best_stereo_model.pth", map_location=device))
    improved_large_epoch_rmse = calculate_rmse(improved_model, val_loader, device)
    print(f"Improved Model with Large Epoch RMSE: {improved_large_epoch_rmse:.4f}")

    # 결과 비교
    improvement1 = ((original_rmse - improved_rmse) / original_rmse) * 100
    improvement2 = ((original_rmse - improved_large_epoch_rmse) / original_rmse) * 100
    print(f"\noriginal -> improved model RMSE Improvement: {improvement1:.2f}%")
    print(f"\noriginal -> imporved model with large epoch RMSE Improvement: {improvement2:.2f}%")
    
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

def apply_gamma_correction(disparity_map, gamma=0.5, percentile_range=(5, 95)):
    """깊이맵에 gamma correction과 percentile clipping 적용"""
    lo, hi = np.percentile(disparity_map, percentile_range[0]), np.percentile(disparity_map, percentile_range[1])
    clipped = np.clip((disparity_map - lo) / (hi - lo + 1e-8), 0, 1)
    gamma_corrected = clipped ** gamma
    return gamma_corrected

def calculate_error_map(pred, gt, valid_mask):
    """예측과 실제값 사이의 오차 맵 계산"""
    error = np.abs(pred - gt)
    error[~valid_mask] = 0  # invalid 픽셀은 0으로 설정
    return error

def visualize_training_samples(original_model, improved_model, improved_large_model, dataset, device, num_samples=5):
    """Training 샘플들에 대한 시각화 (3개 모델 비교)"""
    print("=== Training Samples Visualization ===")
    
    # 모델을 evaluation 모드로 설정
    original_model.eval()
    improved_model.eval()
    improved_large_model.eval()
    
    # 샘플 인덱스 선택 (다양한 케이스를 보기 위해)
    sample_indices = [0, len(dataset)//4, len(dataset)//2, 3*len(dataset)//4, len(dataset)-1]
    sample_indices = sample_indices[:num_samples]
    
    fig, axes = plt.subplots(num_samples, 8, figsize=(24, 3*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            left_img, right_img, gt_disp = dataset[idx]
            
            # 배치 차원 추가
            left_batch = left_img.unsqueeze(0).to(device)
            right_batch = right_img.unsqueeze(0).to(device)
            gt_batch = gt_disp.unsqueeze(0).to(device)
            
            # 예측 수행
            original_pred = original_model(left_batch, right_batch)[0, 0].cpu().numpy()
            improved_pred = improved_model(left_batch, right_batch)[0, 0].cpu().numpy()
            improved_large_pred = improved_large_model(left_batch, right_batch)[0, 0].cpu().numpy()
            gt_np = gt_batch[0, 0].cpu().numpy()
            
            # Valid mask 생성
            valid_mask = gt_np > 0
            
            # 1. Input 이미지
            axes[i, 0].imshow(left_img.permute(1, 2, 0).cpu())
            axes[i, 0].set_title(f"Input {idx}")
            axes[i, 0].axis('off')
            
            # 2. Ground Truth
            gt_gamma = apply_gamma_correction(gt_np)
            axes[i, 1].imshow(gt_gamma, cmap='magma')
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis('off')
            
            # 3. Original Model 예측
            orig_gamma = apply_gamma_correction(original_pred)
            axes[i, 2].imshow(orig_gamma, cmap='magma')
            axes[i, 2].set_title("Original (Simple Conv)")
            axes[i, 2].axis('off')
            
            # 4. Improved Model 예측 (20 epochs)
            imp_gamma = apply_gamma_correction(improved_pred)
            axes[i, 3].imshow(imp_gamma, cmap='magma')
            axes[i, 3].set_title("Improved (ResNet-20ep)")
            axes[i, 3].axis('off')
            
            # 5. Improved Model 예측 (Large epochs)
            imp_large_gamma = apply_gamma_correction(improved_large_pred)
            axes[i, 4].imshow(imp_large_gamma, cmap='magma')
            axes[i, 4].set_title("Improved (ResNet-Large)")
            axes[i, 4].axis('off')
            
            # 6. Original 오차 맵
            orig_error = calculate_error_map(original_pred, gt_np, valid_mask)
            orig_error_gamma = apply_gamma_correction(orig_error, gamma=1.0)
            axes[i, 5].imshow(orig_error_gamma, cmap='hot')
            axes[i, 5].set_title("Original Error")
            axes[i, 5].axis('off')
            
            # 7. Improved 오차 맵 (20 epochs)
            imp_error = calculate_error_map(improved_pred, gt_np, valid_mask)
            imp_error_gamma = apply_gamma_correction(imp_error, gamma=1.0)
            axes[i, 6].imshow(imp_error_gamma, cmap='hot')
            axes[i, 6].set_title("Improved Error (20ep)")
            axes[i, 6].axis('off')
            
            # 8. Improved 오차 맵 (Large epochs)
            imp_large_error = calculate_error_map(improved_large_pred, gt_np, valid_mask)
            imp_large_error_gamma = apply_gamma_correction(imp_large_error, gamma=1.0)
            axes[i, 7].imshow(imp_large_error_gamma, cmap='hot')
            axes[i, 7].set_title("Improved Error (Large)")
            axes[i, 7].axis('off')
            
            # 정량적 지표 계산
            if valid_mask.sum() > 0:
                orig_mae = np.mean(np.abs(original_pred[valid_mask] - gt_np[valid_mask]))
                imp_mae = np.mean(np.abs(improved_pred[valid_mask] - gt_np[valid_mask]))
                imp_large_mae = np.mean(np.abs(improved_large_pred[valid_mask] - gt_np[valid_mask]))
                print(f"Sample {idx} - Original MAE: {orig_mae:.4f}, Improved(20ep) MAE: {imp_mae:.4f}, Improved(Large) MAE: {imp_large_mae:.4f}")
    
    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/training_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Training samples visualization saved to output/training_comparison.png")

def visualize_test_samples(original_model, improved_model, improved_large_model, root_dir, transform, device, num_samples=5):
    """Test 샘플들에 대한 시각화 (Ground Truth 없음, 3개 모델 비교)"""
    print("\n=== Test Samples Visualization ===")
    
    # 모델을 evaluation 모드로 설정
    original_model.eval()
    improved_model.eval()
    improved_large_model.eval()
    
    # Test 이미지 로드
    test_dataset = StereoDepthKITTITest(root_dir=root_dir, transform=transform)
    test_indices = list(range(min(num_samples, len(test_dataset))))
    
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i, idx in enumerate(test_indices):
            left_img, right_img = test_dataset[idx]
            
            # 배치 차원 추가
            left_batch = left_img.unsqueeze(0).to(device)
            right_batch = right_img.unsqueeze(0).to(device)
            
            # 예측 수행
            original_pred = original_model(left_batch, right_batch)[0, 0].cpu().numpy()
            improved_pred = improved_model(left_batch, right_batch)[0, 0].cpu().numpy()
            improved_large_pred = improved_large_model(left_batch, right_batch)[0, 0].cpu().numpy()
            
            # 1. Input 이미지
            axes[i, 0].imshow(left_img.permute(1, 2, 0).cpu())
            axes[i, 0].set_title(f"Test Input {idx}")
            axes[i, 0].axis('off')
            
            # 2. Right 이미지 (참고용)
            axes[i, 1].imshow(right_img.permute(1, 2, 0).cpu())
            axes[i, 1].set_title("Right Image")
            axes[i, 1].axis('off')
            
            # 3. Original Model 예측
            orig_gamma = apply_gamma_correction(original_pred)
            axes[i, 2].imshow(orig_gamma, cmap='magma')
            axes[i, 2].set_title("Original (Simple Conv)")
            axes[i, 2].axis('off')
            
            # 4. Improved Model 예측 (20 epochs)
            imp_gamma = apply_gamma_correction(improved_pred)
            axes[i, 3].imshow(imp_gamma, cmap='magma')
            axes[i, 3].set_title("Improved (ResNet-20ep)")
            axes[i, 3].axis('off')
            
            # 5. Improved Model 예측 (Large epochs)
            imp_large_gamma = apply_gamma_correction(improved_large_pred)
            axes[i, 4].imshow(imp_large_gamma, cmap='magma')
            axes[i, 4].set_title("Improved (ResNet-Large)")
            axes[i, 4].axis('off')
            
            # 예측 통계 출력
            print(f"Test Sample {idx}:")
            print(f"  Original - Min: {original_pred.min():.4f}, Max: {original_pred.max():.4f}, Mean: {original_pred.mean():.4f}")
            print(f"  Improved(20ep) - Min: {improved_pred.min():.4f}, Max: {improved_pred.max():.4f}, Mean: {improved_pred.mean():.4f}")
            print(f"  Improved(Large) - Min: {improved_large_pred.min():.4f}, Max: {improved_large_pred.max():.4f}, Mean: {improved_large_pred.mean():.4f}")
    
    plt.tight_layout()
    plt.savefig("output/test_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Test samples visualization saved to output/test_comparison.png")

def create_difference_visualization(original_model, improved_model, improved_large_model, dataset, device, num_samples=3):
    """세 모델 간의 차이를 강조한 시각화"""
    print("\n=== Model Difference Visualization ===")
    
    original_model.eval()
    improved_model.eval()
    improved_large_model.eval()
    
    # 다양한 난이도의 샘플 선택
    sample_indices = [0, len(dataset)//3, 2*len(dataset)//3][:num_samples]
    
    fig, axes = plt.subplots(num_samples, 7, figsize=(28, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            left_img, right_img, gt_disp = dataset[idx]
            
            left_batch = left_img.unsqueeze(0).to(device)
            right_batch = right_img.unsqueeze(0).to(device)
            gt_batch = gt_disp.unsqueeze(0).to(device)
            
            original_pred = original_model(left_batch, right_batch)[0, 0].cpu().numpy()
            improved_pred = improved_model(left_batch, right_batch)[0, 0].cpu().numpy()
            improved_large_pred = improved_large_model(left_batch, right_batch)[0, 0].cpu().numpy()
            gt_np = gt_batch[0, 0].cpu().numpy()
            
            valid_mask = gt_np > 0
            
            # 1. Input
            axes[i, 0].imshow(left_img.permute(1, 2, 0).cpu())
            axes[i, 0].set_title(f"Input {idx}")
            axes[i, 0].axis('off')
            
            # 2. Ground Truth
            gt_gamma = apply_gamma_correction(gt_np)
            axes[i, 1].imshow(gt_gamma, cmap='magma')
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis('off')
            
            # 3. 차이 맵 1: Improved(20ep) - Original
            diff_map1 = improved_pred - original_pred
            diff_abs1 = np.abs(diff_map1)
            axes[i, 2].imshow(diff_abs1, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
            axes[i, 2].set_title("Diff: |Improved(20ep) - Original|")
            axes[i, 2].axis('off')
            
            # 4. 차이 맵 2: Improved(Large) - Original
            diff_map2 = improved_large_pred - original_pred
            diff_abs2 = np.abs(diff_map2)
            axes[i, 3].imshow(diff_abs2, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
            axes[i, 3].set_title("Diff: |Improved(Large) - Original|")
            axes[i, 3].axis('off')
            
            # 5. 차이 맵 3: Improved(Large) - Improved(20ep)
            diff_map3 = improved_large_pred - improved_pred
            diff_abs3 = np.abs(diff_map3)
            axes[i, 4].imshow(diff_abs3, cmap='RdBu_r', vmin=-0.05, vmax=0.05)
            axes[i, 4].set_title("Diff: |Large - 20ep|")
            axes[i, 4].axis('off')
            
            # 6. 개선된 영역 하이라이트 (Original vs Best)
            orig_error = np.abs(original_pred - gt_np)
            imp_large_error = np.abs(improved_large_pred - gt_np)
            improvement_mask = (orig_error > imp_large_error) & valid_mask
            
            highlight = np.zeros_like(gt_np)
            highlight[improvement_mask] = 1
            axes[i, 5].imshow(highlight, cmap='Greens', alpha=0.7)
            axes[i, 5].imshow(apply_gamma_correction(gt_np), cmap='magma', alpha=0.3)
            axes[i, 5].set_title("Best Improved Regions\n(Green = Better)")
            axes[i, 5].axis('off')
            
            # 7. 정량적 비교 (막대 그래프)
            if valid_mask.sum() > 0:
                orig_mae = np.mean(orig_error[valid_mask])
                imp_mae = np.mean(np.abs(improved_pred[valid_mask] - gt_np[valid_mask]))
                imp_large_mae = np.mean(imp_large_error[valid_mask])
                
                axes[i, 6].bar(['Original', 'Improved(20ep)', 'Improved(Large)'], 
                              [orig_mae, imp_mae, imp_large_mae], 
                              color=['lightcoral', 'lightblue', 'lightgreen'])
                axes[i, 6].set_title(f'MAE Comparison\nBest: {((orig_mae-imp_large_mae)/orig_mae*100):.1f}% improvement')
                axes[i, 6].set_ylabel('Mean Absolute Error')
                axes[i, 6].tick_params(axis='x', rotation=45)
                
                # 개선 비율 출력
                improvement_ratio = improvement_mask.sum() / valid_mask.sum() * 100
                print(f"Sample {idx}: {improvement_ratio:.1f}% of valid pixels improved (Original vs Large)")
    
    plt.tight_layout()
    plt.savefig("output/difference_analysis.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Difference analysis saved to output/difference_analysis.png")

def main():
    """메인 시각화 함수 (3개 모델 비교)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 데이터 준비
    root_dir = '../data_scene_flow'
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # 데이터셋 로드
    dataset = StereoDepthKITTI(root_dir=root_dir, transform=transform)
    
    # 모델 초기화
    original_model = StereoNet(192)
    improved_model = ImprovedStereoNet(192)
    improved_large_model = ImprovedStereoNet(192)  # 더 많은 epoch으로 학습된 모델
    
    # 학습된 모델 가중치 로드
    model_dir = './models'
    original_path = os.path.join(model_dir, 'best_original_model.pth')
    improved_path = os.path.join(model_dir, 'best_improved_model.pth')
    improved_large_path = os.path.join(model_dir, 'best_stereo_model.pth')
    
    try:
        original_model.load_state_dict(torch.load(original_path, map_location=device))
        improved_model.load_state_dict(torch.load(improved_path, map_location=device))
        improved_large_model.load_state_dict(torch.load(improved_large_path, map_location=device))
        print("✅ All model weights loaded successfully!")
        print(f"   - Original model: {original_path}")
        print(f"   - Improved model (20ep): {improved_path}")  
        print(f"   - Improved model (Large ep): {improved_large_path}")
    except FileNotFoundError as e:
        print(f"❌ Model file not found: {e}")
        print("Please run the training code first to generate the .pth files.")
        return
    
    # 모델을 device로 이동
    original_model.to(device)
    improved_model.to(device)
    improved_large_model.to(device)
    
    print("\n" + "="*60)
    print("STEREO DEPTH ESTIMATION - QUALITATIVE ANALYSIS (3 MODELS)")
    print("="*60)
    
    # 1. Training 샘플 시각화 (3개 모델)
    visualize_training_samples(original_model, improved_model, improved_large_model, dataset, device, num_samples=5)
    
    # 2. Test 샘플 시각화 (3개 모델)
    visualize_test_samples(original_model, improved_model, improved_large_model, root_dir, transform, device, num_samples=5)
    
    # 3. 모델 차이 분석 (3개 모델)
    create_difference_visualization(original_model, improved_model, improved_large_model, dataset, device, num_samples=3)
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("Check the output/ directory for saved images:")
    print("- training_comparison.png: Training samples with GT comparison (3 models)")
    print("- test_comparison.png: Test samples prediction comparison (3 models)") 
    print("- difference_analysis.png: Detailed difference analysis (3 models)")
    print("="*60)

if __name__ == "__main__":
    main()