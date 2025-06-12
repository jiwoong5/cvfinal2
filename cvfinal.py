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

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # 초기 블록
        self.initial = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        # 4개의 ResidualBlock 반복, 채널 32 유지, stride=1
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
    return self.conv(cost)  # [B,1,224,224]

class StereoNet(nn.Module):
  def __init__(self, max_disp=192):
    super().__init__()
    self.feat  = FeatureExtractor()
    self.agg2d = CostVolume2DAggregation(max_disp)
  def forward(self, l, r):
    return self.agg2d(self.feat(l), self.feat(r))
  
def calculate_mae(model, dataloader, device):
    model.eval()
    model.to(device)
    total_abs_error = 0.0
    total_pixels = 0
    
    with torch.no_grad():
        for left_img, right_img, gt_disp in dataloader:
            left_img = left_img.to(device)
            right_img = right_img.to(device)
            gt_disp = gt_disp.to(device)
            
            pred_disp = model(left_img, right_img)
            pred_disp = pred_disp.squeeze(1)  # [B, H, W]
            gt_disp = gt_disp.squeeze(1)      # [B, H, W]

            mask = gt_disp > 0  # valid pixel mask (KITTI 같은 경우 0은 invalid)
            abs_error = torch.abs(pred_disp[mask] - gt_disp[mask]).sum()
            total_abs_error += abs_error.item()
            total_pixels += mask.sum().item()

    mae = total_abs_error / total_pixels
    return mae

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device

    #################################################
    ############### HYPERPARAMETERS #################
    #################################################

    #root_dir = '/content/drive/MyDrive/datasets/stereo_kitti'
    root_dir = '../data_scene_flow'

    batch_size = 8
    num_workers = 2

    num_epochs = 200

    #################################################
    #################################################
    #################################################
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ])
    
    dataset = StereoDepthKITTI(root_dir=root_dir, transform=transform)
    stereo_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, persistent_workers=True, pin_memory=True)
    
    '''    
    for left, right, disp in stereo_loader:
        print(f"Left image shape     : {left.shape}")     # (B, 3, 224, 224)
        print(f"Right image shape    : {right.shape}")    # (B, 3, 224, 224)
        print(f"Disparity map shape  : {disp.shape}")     # (B, 1, 224, 224)
        break  # 첫 배치만 보고 종료
    '''
    print(f"Number of Samples: {len(dataset)}")

    model = StereoNet(192).to(device)
    #################################################
    #################### TO DO ######################
    #################################################
    """
    Please fill in the following optimization components:
    1. criterion : Loss function
    2. optimizer : Optimizer
    3. (optional) scheduler : Learning rate scheduler
    Feel free to experiment with any configuration you prefer.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = None
    #################################################
    #################################################
    #################################################
    best_loss = float('inf')
    #model_dir = '/content/drive/MyDrive/models'
    model_dir = './models'
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, 'best_stereo_model.pth')
    '''
    #모델 학습
    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss = 0
        t0 = time.time()

        for l_img, r_img, gt_disp in stereo_loader:
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

        avg_loss = total_loss / len(stereo_loader)

        # save on every epoch if it's the best so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            # print(f"[Epoch {epoch}] New best loss {best_loss:.4f}, model saved.")

        # print progress every 10 epochs
        if epoch % 10 == 0:
            elapsed = time.time() - t0
            print(f"Epoch {epoch}/{num_epochs} Loss: {avg_loss:.4f} Time: {elapsed:.1f}s")
    '''
    #best_path = os.path.join(model_dir, 'best_mono_model.pth')
    best_path = os.path.join(model_dir, 'best_stereo_model.pth')

    '''
    #mae 계산
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.to(device).eval()

    val_mae = calculate_mae(model, stereo_loader, device)
    print(f"Validation MAE: {val_mae:.4f}")
    '''
    '''
    #training 첫 5개 이미지에 대한 깊이맵 생성 및 결과 비교
    model.load_state_dict(torch.load("models/best_stereo_model.pth", map_location=device))
    model.to(device).eval()

    val_loader = DataLoader(dataset, batch_size=5, shuffle=False, num_workers=2, pin_memory=True)

    l_imgs, r_imgs, gt_disps = next(iter(val_loader))
    l_imgs, r_imgs, gt_disps = l_imgs.to(device), r_imgs.to(device), gt_disps.to(device)

    with torch.no_grad():
      preds = model(l_imgs, r_imgs)

    fig, axes = plt.subplots(5, 3, figsize=(9, 15))
    with torch.no_grad():
      for i in range(5):
        axes[i, 0].imshow(l_imgs[i].permute(1, 2, 0).cpu())
        axes[i, 0].set_title("Input")
        axes[i, 0].axis('off')

        gt = gt_disps[i, 0].cpu().numpy()
        lo_gt, hi_gt = np.percentile(gt, 5), np.percentile(gt, 95)
        gt_clip = np.clip((gt - lo_gt) / (hi_gt - lo_gt + 1e-8), 0, 1)
        gt_gamma = gt_clip ** 0.5
        axes[i, 1].imshow(gt_gamma, cmap='magma')
        axes[i, 1].set_title("GT Disparity")
        axes[i, 1].axis('off')

        pred = preds[i, 0].cpu().numpy()
        lo_p, hi_p = np.percentile(pred, 5), np.percentile(pred, 95)
        pred_clip = np.clip((pred - lo_p) / (hi_p - lo_p + 1e-8), 0, 1)
        pred_gamma = pred_clip ** 0.5
        axes[i, 2].imshow(pred_gamma, cmap='magma')
        axes[i, 2].set_title("Pred Disparity")
        axes[i, 2].axis('off')

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/1_7.png")
    '''
    '''
    #testing 첫 5개 이미지에 대한 깊이맵 생성
    model.load_state_dict(torch.load("models/best_stereo_model.pth", map_location=device))
    model.to(device).eval()

    test_dir = os.path.join(root_dir, 'testing', 'image_2')
    all_files = [f for f in os.listdir(test_dir) if f.endswith('_10.png')]
    all_files.sort()
    test_files = all_files[:5]

    fig, axes = plt.subplots(len(test_files), 2, figsize=(8, 4 * len(test_files)))

    with torch.no_grad():
      for i, fname in enumerate(test_files):
        img_l = Image.open(os.path.join(root_dir, 'testing', 'image_2', fname)).convert('RGB')
        img_r = Image.open(os.path.join(root_dir, 'testing', 'image_3', fname)).convert('RGB')
        inp_l = transform(img_l).unsqueeze(0).to(device)
        inp_r = transform(img_r).unsqueeze(0).to(device)

        pred = model(inp_l, inp_r)[0, 0].cpu().numpy()

        # percentile clipping (5–95%) and gamma correction (γ=0.5)
        lo, hi = np.percentile(pred, 5), np.percentile(pred, 95)
        pred_clip = np.clip((pred - lo) / (hi - lo + 1e-8), 0, 1)
        pred_gamma = pred_clip ** 0.5

        axes[i, 0].imshow(inp_l[0].cpu().permute(1, 2, 0).numpy())
        axes[i, 0].set_title(f"Test Input ({fname})")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(pred_gamma, cmap='magma')
        axes[i, 1].set_title(f"Pred Disparity ({fname})")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig("output/1_7_testing.png")
    '''