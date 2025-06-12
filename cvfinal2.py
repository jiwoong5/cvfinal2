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

    # 왼쪽 이미지와 대응하는 Non-Occluded 왼쪽 시차맵
    left_imgs = sorted(glob(os.path.join(root_dir, 'training/image_2/*_10.png')))[:160]
    left_disps = sorted(glob(os.path.join(root_dir, 'training/disp_noc_0/*_10.png')))[:160]

    assert len(left_imgs) == len(left_disps), \
        f"Images and disparity maps count mismatch: {len(left_imgs)} vs {len(left_disps)}"

    self.img_paths = left_imgs
    self.disp_paths = left_disps

    self.transform_img  = transform_img
    self.transform_disp = transform_disp

  def __len__(self):
    return len(self.img_paths)

  def __getitem__(self, idx):
    img = Image.open(self.img_paths[idx]).convert('RGB')
    if self.transform_img:
      img = self.transform_img(img)

    raw = np.array(Image.open(self.disp_paths[idx]), dtype=np.uint16)
    # depth in meters = raw / 256.0; normalize to [0,1] by /80.0
    depth = torch.from_numpy(raw.astype(np.float32) / 256.0 / 80.0).unsqueeze(0)
    if self.transform_disp:
      depth = self.transform_disp(depth)

    return img, depth

class MonoDepthKITTI_Test(Dataset):
  def __init__(self, root_dir, transform_img=None, transform_disp=None):
    super().__init__()

    # 왼쪽 이미지와 Non-Occluded 시차맵의 뒤쪽 40개
    left_imgs = sorted(glob(os.path.join(root_dir, 'training/image_2/*_10.png')))[-40:]
    left_disps = sorted(glob(os.path.join(root_dir, 'training/disp_noc_0/*_10.png')))[-40:]

    assert len(left_imgs) == len(left_disps), \
        f"Images and disparity maps count mismatch: {len(left_imgs)} vs {len(left_disps)}"

    self.img_paths = left_imgs
    self.disp_paths = left_disps

    self.transform_img = transform_img
    self.transform_disp = transform_disp

  def __len__(self):
    return len(self.img_paths)

  def __getitem__(self, idx):
    img = Image.open(self.img_paths[idx]).convert('RGB')
    if self.transform_img:
      img = self.transform_img(img)

    raw = np.array(Image.open(self.disp_paths[idx]), dtype=np.uint16)
    # depth in meters = raw / 256.0; normalize to [0,1] by /80.0
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
        self.act = nn.Identity()

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)               # [B, 64, H, W]
        e2 = self.enc2(self.pool1(e1)) # [B, 128, H/2, W/2]
        e3 = self.enc3(self.pool2(e2)) # [B, 256, H/4, W/4]
        e4 = self.enc4(self.pool3(e3)) # [B, 512, H/8, W/8]

        b  = self.bottleneck(self.pool4(e4)) # [B, 1024, H/16, W/16]

        d4 = self.up4(b)                      # [B, 512, H/8, W/8]
        d4 = torch.cat((d4, e4), dim=1)      # 채널 합쳐서 [B, 1024, H/8, W/8]
        d4 = self.dec4(d4)                    # 다시 [B, 512, H/8, W/8]

        d3 = self.up3(d4)                     # [B, 256, H/4, W/4]
        d3 = torch.cat((d3, e3), dim=1)      # [B, 512, H/4, W/4]
        d3 = self.dec3(d3)                    # [B, 256, H/4, W/4]

        d2 = self.up2(d3)                     # [B, 128, H/2, W/2]
        d2 = torch.cat((d2, e2), dim=1)      # [B, 256, H/2, W/2]
        d2 = self.dec2(d2)                    # [B, 128, H/2, W/2]

        d1 = self.up1(d2)                     # [B, 64, H, W]
        d1 = torch.cat((d1, e1), dim=1)      # [B, 128, H, W]
        d1 = self.dec1(d1)                    # [B, 64, H, W]

        return self.act(self.conv_last(d1))
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device

    #################################################
    ############### HYPERPARAMETERS #################
    #################################################

    #root_dir = '/content/drive/MyDrive/datasets/mono_kitti'
    root_dir = '../data_scene_flow'

    batch_size = 8
    num_workers = 2

    num_epochs = 300

    #################################################
    #################################################
    #################################################
    transform_img = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    dataset = MonoDepthKITTI(root_dir=root_dir, transform_img=transform_img, transform_disp=transform_disp)
    mono_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True, pin_memory=True)
    print(f"Number of Samples: {len(dataset)}")

    model = UNetDepth().to(device)

    #################################################
    #################### TO DO ######################
    #################################################

    # 1. Loss function (Smooth L1 Loss = Huber Loss, 안정적이고 outlier에 덜 민감)
    criterion = nn.SmoothL1Loss()

    # 2. Optimizer (Adam: 학습 속도 빠르고 일반적으로 잘 작동)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # 3. (Optional) Learning Rate Scheduler (CosineAnnealing은 부드럽게 감소)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

    #################################################
    #################################################
    #################################################
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, 'best_mono_model.pth')

    step = 0
    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss = 0
        t0 = time.time()

        for img, gt_depth in mono_loader:
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

        avg_loss = total_loss / len(mono_loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            # print(f"[Epoch {epoch}] New best loss {best_loss:.4f}, model saved.")

        if epoch % 10 == 0:
            elapsed = time.time() - t0
            print(f"Epoch {epoch}/{num_epochs}  Loss: {avg_loss:.4f}  Time: {elapsed:.1f}s")