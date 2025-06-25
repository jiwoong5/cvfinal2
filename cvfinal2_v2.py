import os, time
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class MonoDepthKITTI(Dataset):
    def __init__(self, root_dir, transform_img=None, transform_disp=None):
        super().__init__()

        # 왼쪽 이미지와 시차맵 (앞 160개)
        left_imgs = sorted(glob(os.path.join(root_dir, 'training/image_2/*_10.png')))[:160]
        left_disps = sorted(glob(os.path.join(root_dir, 'training/disp_noc_0/*_10.png')))[:160]

        # 오른쪽 이미지와 시차맵 (앞 160개)
        right_imgs = sorted(glob(os.path.join(root_dir, 'training/image_3/*_10.png')))[:160]
        right_disps = sorted(glob(os.path.join(root_dir, 'training/disp_noc_1/*_10.png')))[:160]

        assert len(left_imgs) == len(left_disps), \
            f"Left: {len(left_imgs)} imgs vs {len(left_disps)} disps"
        assert len(right_imgs) == len(right_disps), \
            f"Right: {len(right_imgs)} imgs vs {len(right_disps)} disps"

        # 왼쪽 + 오른쪽 합치기
        self.img_paths = left_imgs + right_imgs
        self.disp_paths = left_disps + right_disps

        self.transform_img = transform_img
        self.transform_disp = transform_disp

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        if self.transform_img:
            img = self.transform_img(img)

        raw = np.array(Image.open(self.disp_paths[idx]), dtype=np.uint16)
        depth = torch.from_numpy(raw.astype(np.float32) / 256.0 / 80.0).unsqueeze(0)
        if self.transform_disp:
            depth = self.transform_disp(depth)

        return img, depth


class MonoDepthKITTI_Test(Dataset):
    def __init__(self, root_dir, transform_img=None, transform_disp=None):
        super().__init__()

        # 왼쪽 이미지와 시차맵 (뒤 40개)
        left_imgs = sorted(glob(os.path.join(root_dir, 'training/image_2/*_10.png')))[-40:]
        left_disps = sorted(glob(os.path.join(root_dir, 'training/disp_noc_0/*_10.png')))[-40:]

        # 오른쪽 이미지와 시차맵 (뒤 40개)
        right_imgs = sorted(glob(os.path.join(root_dir, 'training/image_3/*_10.png')))[-40:]
        right_disps = sorted(glob(os.path.join(root_dir, 'training/disp_noc_1/*_10.png')))[-40:]

        assert len(left_imgs) == len(left_disps), \
            f"Left: {len(left_imgs)} imgs vs {len(left_disps)} disps"
        assert len(right_imgs) == len(right_disps), \
            f"Right: {len(right_imgs)} imgs vs {len(right_disps)} disps"

        self.img_paths = left_imgs + right_imgs
        self.disp_paths = left_disps + right_disps

        self.transform_img = transform_img
        self.transform_disp = transform_disp

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        if self.transform_img:
            img = self.transform_img(img)

        raw = np.array(Image.open(self.disp_paths[idx]), dtype=np.uint16)
        depth = torch.from_numpy(raw.astype(np.float32) / 256.0 / 80.0).unsqueeze(0)
        if self.transform_disp:
            depth = self.transform_disp(depth)

        return img, depth


def transform_disp(depth_tensor):
    return F.interpolate(
        depth_tensor.unsqueeze(0),
        size=(224,224),
        mode='nearest'
    ).squeeze(0)

# UNet base
class UNetDepthBase(nn.Module):
    def __init__(self, in_channels=3, features=[64,128,256,512], use_skip=True, use_bn=False):
        super().__init__()
        self.use_skip = use_skip
        self.features = features
        self.enc1 = self._conv_block(in_channels, features[0], use_bn)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._conv_block(features[0], features[1], use_bn)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._conv_block(features[1], features[2], use_bn)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = self._conv_block(features[2], features[3], use_bn)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = self._make_bottleneck(features[3], use_bn)

        self.up4 = nn.ConvTranspose2d(features[3]*2, features[3], 2, 2)
        self.dec4 = self._conv_block(features[3]*2 if use_skip else features[3], features[3], use_bn)

        self.up3 = nn.ConvTranspose2d(features[3], features[2], 2, 2)
        self.dec3 = self._conv_block(features[2]*2 if use_skip else features[2], features[2], use_bn)

        self.up2 = nn.ConvTranspose2d(features[2], features[1], 2, 2)
        self.dec2 = self._conv_block(features[1]*2 if use_skip else features[1], features[1], use_bn)

        self.up1 = nn.ConvTranspose2d(features[1], features[0], 2, 2)
        self.dec1 = self._conv_block(features[0]*2 if use_skip else features[0], features[0], use_bn)

        self.conv_last = nn.Conv2d(features[0], 1, 1)
        self.act = nn.ReLU()

    def _conv_block(self, in_ch, out_ch, use_bn):
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        ]
        return nn.Sequential(*layers)

    def _make_bottleneck(self, in_ch, use_bn):
        return self._conv_block(in_ch, in_ch * 2, use_bn)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat((d4, e4), dim=1) if self.use_skip else d4)

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat((d3, e3), dim=1) if self.use_skip else d3)

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat((d2, e2), dim=1) if self.use_skip else d2)

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat((d1, e1), dim=1) if self.use_skip else d1)

        return self.act(self.conv_last(d1))

# Original UNet
class UNetDepth(UNetDepthBase):
    def __init__(self, in_channels=3, features=[64,128,256,512]):
        super().__init__(in_channels, features, use_skip=True, use_bn=False)

# NoSkip UNet
class UNetDepth_NoSkip(UNetDepthBase):
    def __init__(self, in_channels=3, features=[64,128,256,512]):
        super().__init__(in_channels, features, use_skip=False, use_bn=False)

# BN UNet
class UNetDepth_BN(UNetDepthBase):
    def __init__(self, in_channels=3, features=[64,128,256,512]):
        super().__init__(in_channels, features, use_skip=True, use_bn=True)


def count_parameters(model):
    """모델의 전체 파라미터 수를 계산"""
    return sum(p.numel() for p in model.parameters())


def evaluate_model_metrics(model, dataloader, device):
    """Calculate both MAE and RMSE"""
    model.to(device)
    model.eval()
    total_mae = 0.0
    total_mse = 0.0
    total_pixels = 0
    
    with torch.no_grad():
        for img, depth_gt in dataloader:
            img = img.to(device)
            depth_gt = depth_gt.to(device)
            pred = model(img)
            
            # MAE
            mae = torch.abs(pred - depth_gt)
            total_mae += mae.sum().item()
            
            # MSE for RMSE
            mse = (pred - depth_gt) ** 2
            total_mse += mse.sum().item()
            
            total_pixels += mae.numel()
    
    mae_avg = total_mae / total_pixels
    rmse_avg = np.sqrt(total_mse / total_pixels)
    
    return mae_avg, rmse_avg


def train_model(model, model_name, train_loader, device, num_epochs=300):
    """Train a model and save the best version"""
    print(f"\n=== Training {model_name} ===")
    
    # Loss function and optimizer
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
    
    # Create model directory
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, f'best_mono_{model_name}_model.pth')
    
    step = 0
    best_loss = float('inf')
    
    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss = 0
        t0 = time.time()

        for img, gt_depth in train_loader:
            img, gt_depth = img.to(device), gt_depth.to(device)

            optimizer.zero_grad()
            pred = model(img).squeeze(1)
            gt   = gt_depth.squeeze(1)

            loss = criterion(pred, gt)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)

        if epoch % 50 == 0:
            elapsed = time.time() - t0
            print(f"Epoch {epoch}/{num_epochs}  Loss: {avg_loss:.4f}  Time: {elapsed:.1f}s")
    
    print(f"Training completed. Best loss: {best_loss:.4f}")
    return save_path


def compare_models_performance(model_paths, test_loader, device):
    """Compare performance of different models and return detailed results"""
    print("\n=== Model Performance Comparison ===")
    results = []
    
    for model_path, model_class, model_name in model_paths:
        model = model_class().to(device)
        
        # 모델 파일 존재 확인
        if not os.path.exists(model_path):
            print(f"Warning: Model file {model_path} not found. Skipping {model_name}")
            continue
            
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            mae, rmse = evaluate_model_metrics(model, test_loader, device)
            total_params = count_parameters(model)

            result = {
                'Model': model_name,
                'Parameters': f"{total_params:,}",
                'Parameters_M': f"{total_params/1e6:.2f}M",
                'MAE': f"{mae:.4f}",
                'RMSE': f"{rmse:.4f}",
                'MAE_float': mae,
                'RMSE_float': rmse,
                'Params_count': total_params
            }
            results.append(result)

            print(f"{model_name}:")
            print(f"  Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
            print(f"  MAE:  {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
    
    # Create and save performance table
    if results:
        df = pd.DataFrame(results)
        print("\n=== Performance Summary Table ===")
        print(df[['Model', 'Parameters_M', 'MAE', 'RMSE']].to_string(index=False))
        
        # Save to CSV
        os.makedirs("output", exist_ok=True)
        df[['Model', 'Parameters', 'MAE', 'RMSE']].to_csv("output/model_performance.csv", index=False)
        print("\nPerformance table saved to output/model_performance.csv")
    
    return results


def visualize_depth_comparison(model_paths, test_loader, device, num_samples=5):
    """Visualize depth predictions from all models with random samples"""
    print("\n=== Creating depth map comparison visualization ===")
    
    # Load all models
    models = {}
    for (model_path, model_class, model_name) in model_paths:
        if not os.path.exists(model_path):
            print(f"Warning: Model file {model_path} not found. Skipping {model_name}")
            continue
            
        try:
            model = model_class().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            models[model_name] = model
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
    
    if not models:
        print("No models loaded. Cannot create visualization.")
        return
    
    # Get all test data
    all_imgs = []
    all_gt_depths = []
    for imgs, gt_depths in test_loader:
        all_imgs.append(imgs)
        all_gt_depths.append(gt_depths)
    
    if not all_imgs:
        print("No test data available.")
        return
    
    # Concatenate all batches
    all_imgs = torch.cat(all_imgs, dim=0)
    all_gt_depths = torch.cat(all_gt_depths, dim=0)
    
    # Randomly select samples
    total_samples = len(all_imgs)
    random_indices = torch.randperm(total_samples)[:num_samples]
    
    selected_imgs = all_imgs[random_indices].to(device)
    selected_gt_depths = all_gt_depths[random_indices].to(device)
    
    print(f"Selected {num_samples} random samples from {total_samples} total samples")
    
    # Inverse normalization for visualization
    inv_norm = transforms.Normalize(
        mean=[-m/s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
        std=[1/s for s in [0.229, 0.224, 0.225]]
    )
    
    # Create visualization
    num_models = len(models)
    fig, axes = plt.subplots(num_samples, 2 + num_models, figsize=(5*(2+num_models), 4*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i in range(num_samples):
            # Original image
            img_denorm = inv_norm(selected_imgs[i].cpu())
            img_vis = img_denorm.permute(1, 2, 0).numpy()
            img_vis = np.clip(img_vis, 0, 1)
            axes[i, 0].imshow(img_vis)
            axes[i, 0].set_title(f"Input Image {random_indices[i].item()}", fontsize=12)
            axes[i, 0].axis('off')
            
            # Ground truth
            gt_np = selected_gt_depths[i, 0].cpu().numpy()
            # 유효한 값들만 필터링
            valid_mask = gt_np > 0
            # 수정 (gamma 값 적용을 위해)
            if valid_mask.sum() > 0:
                lo_gt, hi_gt = np.percentile(gt_np[valid_mask], [5, 95])
                gt_clip = np.clip((gt_np - lo_gt) / (hi_gt - lo_gt + 1e-8), 0, 1)
                gt_gamma = gt_clip ** 0.5
            else:
                gt_gamma = np.zeros_like(gt_np)  # 
            axes[i, 1].imshow(gt_gamma, cmap='magma')
            axes[i, 1].set_title("Ground Truth", fontsize=12)
            axes[i, 1].axis('off')
            
            # Model predictions
            col_idx = 2
            for model_name, model in models.items():
                try:
                    pred = model(selected_imgs[i:i+1])[0, 0].cpu().numpy()
                    valid_pred = pred > 0
                    if valid_pred.sum() > 0:
                        lo_p, hi_p = np.percentile(pred[valid_pred], [5, 95])
                        pred_clip = np.clip((pred - lo_p) / (hi_p - lo_p + 1e-8), 0, 1)
                        pred_gamma = pred_clip ** 0.5
                    else:
                        pred_gamma = pred
                    axes[i, col_idx].imshow(pred_gamma, cmap='magma')
                    axes[i, col_idx].set_title(f"{model_name}", fontsize=12)
                    axes[i, col_idx].axis('off')
                    col_idx += 1
                except Exception as e:
                    print(f"Error predicting with {model_name}: {e}")
    
    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/random_samples_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Random samples comparison visualization saved to output/random_samples_comparison.png")

if __name__ == "__main__":
    # 데이터 변환 설정
    transform_img = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 데이터 로더 설정
    ROOT_DIR = "../data_scene_flow"  # KITTI 데이터셋 경로로 수정
    train_dataset = MonoDepthKITTI(ROOT_DIR, transform_img=transform_img, transform_disp=transform_disp)
    test_dataset = MonoDepthKITTI_Test(ROOT_DIR, transform_img=transform_img, transform_disp=transform_disp)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 모델 정의
    models_to_train = [
        (UNetDepth(), "UNet"),
        (UNetDepth_NoSkip(), "UNet_NoSkip"),
        (UNetDepth_BN(), "UNet_BN")
    ]
    '''
    # 모델 훈련
    trained_model_paths = []
    for model, name in models_to_train:
        model_path = train_model(model, name, train_loader, device, num_epochs=20)
        trained_model_paths.append((model_path, type(model), name))
    '''
    trained_model_paths = "./best_mono_model"
    # 모델 성능 비교
    results = compare_models_performance(trained_model_paths, test_loader, device)
    
    # 깊이맵 시각화 비교 (5개 랜덤 샘플)
    visualize_depth_comparison(trained_model_paths, test_loader, device, num_samples=5)