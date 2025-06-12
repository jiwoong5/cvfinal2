import os, time
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

class MonoDepthKITTI(Dataset):
  def __init__(self, root_dir, transform_img=None, transform_disp=None):
    super().__init__()
    self.img_paths  = sorted(glob(os.path.join(root_dir, 'training/image_2/*_10.png')))
    self.disp_paths  = sorted(glob(os.path.join(root_dir, 'training/disp_occ_0/*.png')))
    assert len(self.img_paths) == len(self.disp_paths)
    self.transform_img  = transform_img
    self.transform_disp = transform_disp

  def __len__(self):
    return len(self.img_paths)

  def __getitem__(self, idx):
    img = Image.open(self.img_paths[idx]).convert('RGB')
    if self.transform_img:
      img = self.transform_img(img)

    raw = np.array(Image.open(self.disp_paths[idx]), dtype=np.uint16)
    # depth in meters = raw/256.0; now normalize to [0,1] by /80.0
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