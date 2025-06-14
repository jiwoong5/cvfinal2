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


# Original UNet
class UNetDepth(nn.Module):
    def __init__(self, in_channels=3, features=[64,128,256,512]):
        super().__init__()
        self.enc1 = self._conv_block(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._conv_block(features[0], features[1])
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._conv_block(features[1], features[2])
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = self._conv_block(features[2], features[3])
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = self._conv_block(features[3], features[3]*2)

        # Decoder: 인코더 채널을 거꾸로 하여 ConvTranspose2d와 _conv_block 구성
        self.up4  = nn.ConvTranspose2d(features[3]*2, features[3], kernel_size=2, stride=2)
        self.dec4 = self._conv_block(features[3]*2, features[3])
        
        self.up3  = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.dec3 = self._conv_block(features[2]*2, features[2])
        
        self.up2  = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.dec2 = self._conv_block(features[1]*2, features[1])
        
        self.up1  = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.dec1 = self._conv_block(features[0]*2, features[0])

        self.conv_last = nn.Conv2d(features[0], 1, 1)
        self.act = nn.ReLU()  # Identity -> ReLU로 변경

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)               # [B, 64, H, W]
        e2 = self.enc2(self.pool1(e1))  # [B, 128, H/2, W/2]
        e3 = self.enc3(self.pool2(e2))  # [B, 256, H/4, W/4]
        e4 = self.enc4(self.pool3(e3))  # [B, 512, H/8, W/8]

        b  = self.bottleneck(self.pool4(e4))  # [B, 1024, H/16, W/16]

        d4 = self.up4(b)                      # [B, 512, H/8, W/8]
        d4 = torch.cat((d4, e4), dim=1)       # 채널 합쳐서 [B, 1024, H/8, W/8]
        d4 = self.dec4(d4)                    # 다시 [B, 512, H/8, W/8]

        d3 = self.up3(d4)                     # [B, 256, H/4, W/4]
        d3 = torch.cat((d3, e3), dim=1)       # [B, 512, H/4, W/4]
        d3 = self.dec3(d3)                    # [B, 256, H/4, W/4]

        d2 = self.up2(d3)                     # [B, 128, H/2, W/2]
        d2 = torch.cat((d2, e2), dim=1)       # [B, 256, H/2, W/2]
        d2 = self.dec2(d2)                    # [B, 128, H/2, W/2]

        d1 = self.up1(d2)                     # [B, 64, H, W]
        d1 = torch.cat((d1, e1), dim=1)       # [B, 128, H, W]
        d1 = self.dec1(d1)                    # [B, 64, H, W]

        return self.act(self.conv_last(d1))


# UNet without Skip Connections
class UNetDepth_NoSkip(nn.Module):
    def __init__(self, in_channels=3, features=[70, 140, 280, 560]):
        super().__init__()
        self.enc1 = self._conv_block(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._conv_block(features[0], features[1])
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._conv_block(features[1], features[2])
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = self._conv_block(features[2], features[3])
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = self._conv_block(features[3], features[3]*2)
        self.up4 = nn.ConvTranspose2d(features[3]*2, features[3], kernel_size=2, stride=2)
        self.dec4 = self._conv_block(features[3], features[3])
        self.up3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.dec3 = self._conv_block(features[2], features[2])
        self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.dec2 = self._conv_block(features[1], features[1])
        self.up1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.dec1 = self._conv_block(features[0], features[0])
        self.conv_last = nn.Conv2d(features[0], 1, 1)
        self.act = nn.ReLU()
        
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b  = self.bottleneck(self.pool4(e4))
        d4 = self.dec4(self.up4(b))
        d3 = self.dec3(self.up3(d4))
        d2 = self.dec2(self.up2(d3))
        d1 = self.dec1(self.up1(d2))
        return self.act(self.conv_last(d1))


# UNet with Batch Normalization
class UNetDepth_BN(nn.Module):
    def __init__(self, in_channels=3, features=[64,128,256,512]):
        super().__init__()
        self.enc1 = self._conv_block(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._conv_block(features[0], features[1])
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._conv_block(features[1], features[2])
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = self._conv_block(features[2], features[3])
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = self._conv_block(features[3], features[3]*2)
        self.up4 = nn.ConvTranspose2d(features[3]*2, features[3], 2, 2)
        self.dec4 = self._conv_block(features[3]*2, features[3])
        self.up3 = nn.ConvTranspose2d(features[3], features[2], 2, 2)
        self.dec3 = self._conv_block(features[2]*2, features[2])
        self.up2 = nn.ConvTranspose2d(features[2], features[1], 2, 2)
        self.dec2 = self._conv_block(features[1]*2, features[1])
        self.up1 = nn.ConvTranspose2d(features[1], features[0], 2, 2)
        self.dec1 = self._conv_block(features[0]*2, features[0])
        self.conv_last = nn.Conv2d(features[0], 1, 1)
        self.act = nn.ReLU()
        
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
        d4 = self.dec4(torch.cat((self.up4(b), e4), dim=1))
        d3 = self.dec3(torch.cat((self.up3(d4), e3), dim=1))
        d2 = self.dec2(torch.cat((self.up2(d3), e2), dim=1))
        d1 = self.dec1(torch.cat((self.up1(d2), e1), dim=1))
        return self.act(self.conv_last(d1))


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


def compare_models_performance(model_paths, model_classes, test_loader, device):
    """Compare performance of different models"""
    print("\n=== Model Performance Comparison ===")
    results = {}
    
    for i, (model_path, model_class, model_name) in enumerate(model_paths):
        model = model_class().to(device)
        
        # 모델 파일 존재 확인
        if not os.path.exists(model_path):
            print(f"Warning: Model file {model_path} not found. Skipping {model_name}")
            continue
            
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            mae, rmse = evaluate_model_metrics(model, test_loader, device)
            results[model_name] = {'MAE': mae, 'RMSE': rmse}
            total_params = sum(p.numel() for p in model.parameters())

            print(f"{model_name}:")
            print(f"  MAE:  {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"전체 파라미터 수: {total_params:,}")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
    
    return results


def visualize_depth_comparison(model_paths, test_loader, device, num_samples=3):
    """Visualize depth predictions from all three models"""
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
    
    # Get test samples
    try:
        imgs, gt_depths = next(iter(test_loader))
        imgs, gt_depths = imgs.to(device), gt_depths.to(device)
    except Exception as e:
        print(f"Error loading test data: {e}")
        return
    
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
            img_denorm = inv_norm(imgs[i].cpu())
            img_vis = img_denorm.permute(1, 2, 0).numpy()
            img_vis = np.clip(img_vis, 0, 1)
            axes[i, 0].imshow(img_vis)
            axes[i, 0].set_title("Input Image")
            axes[i, 0].axis('off')
            
            # Ground truth
            gt_np = gt_depths[i, 0].cpu().numpy()
            # 유효한 값들만 필터링
            valid_mask = gt_np > 0
            if valid_mask.sum() > 0:
                lo_gt, hi_gt = np.percentile(gt_np[valid_mask], [5, 95])
                gt_clip = np.clip((gt_np - lo_gt) / (hi_gt - lo_gt + 1e-8), 0, 1)
                gt_gamma = gt_clip ** 0.5
            else:
                gt_gamma = gt_np
            axes[i, 1].imshow(gt_gamma, cmap='magma')
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis('off')
            
            # Model predictions
            col_idx = 2
            for model_name, model in models.items():
                try:
                    pred = model(imgs[i:i+1])[0, 0].cpu().numpy()
                    valid_pred = pred > 0
                    if valid_pred.sum() > 0:
                        lo_p, hi_p = np.percentile(pred[valid_pred], [5, 95])
                        pred_clip = np.clip((pred - lo_p) / (hi_p - lo_p + 1e-8), 0, 1)
                        pred_gamma = pred_clip ** 0.5
                    else:
                        pred_gamma = pred
                    axes[i, col_idx].imshow(pred_gamma, cmap='magma')
                    axes[i, col_idx].set_title(f"{model_name}")
                    axes[i, col_idx].axis('off')
                    col_idx += 1
                except Exception as e:
                    print(f"Error predicting with {model_name}: {e}")
    
    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/model_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Comparison visualization saved to output/model_comparison.png")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #################################################
    ############### HYPERPARAMETERS #################
    #################################################
    root_dir = '../data_scene_flow'
    batch_size = 8
    num_workers = 2
    num_epochs = 20  # Reduced for faster training

    #################################################
    ############### DATA PREPARATION ################
    #################################################
    transform_img = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    # Check if data directory exists
    if not os.path.exists(root_dir):
        print(f"Error: Data directory {root_dir} not found!")
        print("Please check the path or create sample data for testing.")
        exit(1)

    try:
        # Training dataset
        train_dataset = MonoDepthKITTI(root_dir=root_dir, transform_img=transform_img, transform_disp=transform_disp)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                 num_workers=num_workers, persistent_workers=True, pin_memory=True)
        print(f"Training samples: {len(train_dataset)}")

        # Test dataset
        test_dataset = MonoDepthKITTI_Test(root_dir=root_dir, transform_img=transform_img, transform_disp=transform_disp)
        test_loader = DataLoader(test_dataset, batch_size=batch_size//2, shuffle=False, 
                                num_workers=num_workers, persistent_workers=True, pin_memory=True)
        print(f"Test samples: {len(test_dataset)}")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        exit(1)

    #################################################
    ############### MODEL TRAINING ##################
    #################################################
    
    # Define models to train
    models_to_train = [
        #(UNetDepth(), "original"),
        (UNetDepth_NoSkip(), "noskip"),
        #(UNetDepth_BN(), "batch_norm")
    ]
    
    # Uncomment to train models
    # Train all models
    trained_model_paths = []
    for model, model_name in models_to_train:
        model = model.to(device)
        model_path = train_model(model, model_name, train_loader, device, num_epochs)
        trained_model_paths.append((model_path, type(model), model_name))
    
    # 모델 경로 직접 지정
    model_paths = [
        "./models/best_mono_original_model.pth",
        "./models/best_mono_noskip_model.pth",
        "./models/best_mono_batch_norm_model.pth"
    ]

    trained_model_paths = [
        (model_paths[0], UNetDepth, "original"),
        (model_paths[1], UNetDepth_NoSkip, "noskip"),
        (model_paths[2], UNetDepth_BN, "batch_norm"),
    ]

    #################################################
    ############### MODEL EVALUATION ################
    #################################################
    
    # Compare model performance
    results = compare_models_performance(trained_model_paths, [UNetDepth, UNetDepth_NoSkip, UNetDepth_BN], test_loader, device)

    if not results:
        print("No models could be evaluated. Please check model files.")
        exit(1)
    
    # Create performance comparison plot
    model_names = list(results.keys())
    mae_values = [results[name]['MAE'] for name in model_names]
    rmse_values = [results[name]['RMSE'] for name in model_names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # MAE comparison
    colors = ['blue', 'orange', 'green'][:len(model_names)]
    bars1 = ax1.bar(model_names, mae_values, color=colors, alpha=0.7)
    ax1.set_ylabel('MAE')
    ax1.set_title('Mean Absolute Error Comparison')
    ax1.set_ylim(0, max(mae_values) * 1.1)
    for bar, val in zip(bars1, mae_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mae_values)*0.01, 
                f'{val:.4f}', ha='center', va='bottom')
    
    # RMSE comparison
    bars2 = ax2.bar(model_names, rmse_values, color=colors, alpha=0.7)
    ax2.set_ylabel('RMSE')
    ax2.set_title('Root Mean Square Error Comparison')
    ax2.set_ylim(0, max(rmse_values) * 1.1)
    for bar, val in zip(bars2, rmse_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmse_values)*0.01, 
                f'{val:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/performance_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()

    #################################################
    ############ DEPTH MAP VISUALIZATION ############
    #################################################
    
    # Visualize depth map comparisons
    visualize_depth_comparison(trained_model_paths, test_loader, device, num_samples=5)
    
    print("\n=== Training and Evaluation Complete ===")