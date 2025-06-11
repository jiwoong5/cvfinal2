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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

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
    dataset = StereoDepthKITTI(root_dir=root_dir, transform=transform)
    stereo_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, persistent_workers=True, pin_memory=True)

    for left, right, disp in stereo_loader:
        print(f"Left image shape     : {left.shape}")     # (B, 3, 224, 224)
        print(f"Right image shape    : {right.shape}")    # (B, 3, 224, 224)
        print(f"Disparity map shape  : {disp.shape}")     # (B, 1, 224, 224)
        break  # 첫 배치만 보고 종료

    print(f"Number of Samples: {len(dataset)}")