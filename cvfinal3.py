import os
import time
import math
import random
import warnings
from glob import glob
from collections import Counter

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms
import torchvision.ops as ops
from torchvision.ops import box_iou

from sklearn.metrics import average_precision_score

class DetectKITTI(Dataset):
  def __init__(self, root_dir, img_size=(416, 416), flip_prob=0.5):
    self.img_size = img_size
    self.flip_prob = flip_prob
    base = root_dir
    self.images = sorted(glob(os.path.join(base, 'data_object_image_2/training/image_2', '*.png')))  # list all image file paths
    self.labels = sorted(glob(os.path.join(base, 'training/label_2', '*.txt')))  # list all label file paths
    assert len(self.images) == len(self.labels)

    self.transform = transforms.Compose([
      transforms.Resize(self.img_size),  # resize to (416, 416)
      transforms.ToTensor(),
      transforms.Normalize(
          mean = [0.485, 0.456, 0.406],
          std  = [0.229, 0.224, 0.225]
      ),
    ])

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    # load image and get original size
    img = Image.open(self.images[idx]).convert('RGB')
    orig_w, orig_h = img.size

    boxes, labels = [], []
    # read and parse label file
    with open(self.labels[idx], 'r') as f:
      for line in f:
        toks = line.split()
        cls = toks[0]
        if cls not in CLASS_MAP:
          continue
        x1, y1, x2, y2 = map(float, toks[4:8])  # extract bounding box coords
        boxes.append([x1, y1, x2, y2])
        labels.append(CLASS_MAP[cls])

    boxes = np.array(boxes, dtype=np.float32)
    if boxes.size == 0:
      boxes = np.zeros((0, 4), dtype=np.float32)
      labels = []

    # apply random horizontal flip
    if random.random() < self.flip_prob:
      img = img.transpose(Image.FLIP_LEFT_RIGHT)
      if boxes.shape[0] > 0:
        x1 = orig_w - boxes[:, 2]
        x2 = orig_w - boxes[:, 0]
        boxes[:, 0], boxes[:, 2] = x1, x2  # update box x-coordinates

    img_t = self.transform(img)  # resize and normalize image

    # scale boxes to resized image dimensions
    new_w, new_h = self.img_size
    sx, sy = new_w / orig_w, new_h / orig_h
    boxes[:, [0, 2]] *= sx  # scale x
    boxes[:, [1, 3]] *= sy  # scale y

    return img_t, {
      'boxes': torch.from_numpy(boxes),
      'labels': torch.tensor(labels, dtype=torch.int64),
      'orig_size': (orig_w, orig_h)
    }

def yolo_loss(out1, out2, targets):
 losses = {'xy': 0, 'wh': 0, 'obj': 0, 'cls': 0}
 for i, pred in enumerate([out1, out2]):
  B, _, H, W = pred.shape
  # reshape to (B, anchors, H, W, 5 + num_classes)
  pred = pred.view(B, 3, 5 + num_classes, H, W).permute(0, 1, 3, 4, 2)

  # prepare target tensor
  ttarget = torch.zeros_like(pred)

  for b in range(B):
   boxes  = targets[b]['boxes']
   labels = targets[b]['labels']
   for box, cls in zip(boxes, labels):
    x1, y1, x2, y2 = box
    # compute center and size in grid units
    cx = (x1 + x2) / 2 / strides[i]
    cy = (y1 + y2) / 2 / strides[i]
    gw = (x2 - x1) / strides[i]
    gh = (y2 - y1) / strides[i]

    # determine grid cell indices
    gi = min(int(cx), W - 1)
    gj = min(int(cy), H - 1)

    # select best-matching anchor
    anchor_list = anchors_big if i == 0 else anchors_small
    ratios = [(gw / aw, gh / ah) for aw, ah in anchor_list]
    idx    = max(range(3), key=lambda a: min(ratios[a]))

    # assign target values
    ttarget[b, idx, gj, gi, 0]      = cx - gi
    ttarget[b, idx, gj, gi, 1]      = cy - gj
    ttarget[b, idx, gj, gi, 2]      = torch.log(gw / anchor_list[idx][0] + 1e-16)
    ttarget[b, idx, gj, gi, 3]      = torch.log(gh / anchor_list[idx][1] + 1e-16)
    ttarget[b, idx, gj, gi, 4]      = 1.0  # objectness
    ttarget[b, idx, gj, gi, 5 + cls] = 1.0  # one-hot class

  # split pred and target tensors
  pxy, txy   = pred[..., 0:2],  ttarget[..., 0:2]
  pwh, twh   = pred[..., 2:4],  ttarget[..., 2:4]
  pobj, tobj = pred[..., 4],    ttarget[..., 4]
  pcls, tcls = pred[..., 5:],   ttarget[..., 5:]

  obj_mask = tobj == 1  # positive object locations

  # compute losses for positive locations
  losses['xy']  += mse_loss(pxy[obj_mask],  txy[obj_mask])
  losses['wh']  += mse_loss(pwh[obj_mask],  twh[obj_mask])
  losses['obj'] += bce_loss(pobj, tobj)
  losses['cls'] += bce_loss(pcls[obj_mask], tcls[obj_mask])

 total_loss = losses['xy'] + losses['wh'] + losses['obj'] + losses['cls']
 return total_loss / out1.shape[0]  # average over batch

def yolo_collate_fn(batch):
  # collate function to stack image tensors and gather targets
  imgs, targets = zip(*batch)
  return torch.stack(imgs), list(targets)


# KITTI class names
KITTI_CLASSES = [
    'Car', 'Van', 'Truck', 'Pedestrian',
    'Person_sitting', 'Cyclist', 'Tram', 'Misc'
]
CLASS_MAP = {c: i for i, c in enumerate(KITTI_CLASSES)}

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device

    #################################################
    ############### HYPERPARAMETERS #################
    #################################################

    #root_dir = '../data_scene_flow'

    batch_size = 32
    num_workers = 2

    num_epochs = 200

    #################################################
    #################################################
    #################################################

    # Visualization color map (RGB)
    orig_color_map = {
        'Car': (255, 0, 0),
        'Van': (255, 128, 0),
        'Truck': (200, 200, 0), 
        'Pedestrian': (0, 255, 0),
        'Person_sitting': (0, 200, 200),
        'Cyclist': (0, 0, 255),
        'Tram': (128, 0, 255),
        'Misc': (255, 0, 255)
    }
    class_color_map = { CLASS_MAP[c]: orig_color_map[c] for c in KITTI_CLASSES }

    base = os.path.join("../")
    image_paths = sorted(glob(os.path.join(base, 'data_object_image_2/training/image_2', '*.png')))
    label_paths = sorted(glob(os.path.join(base, 'training/label_2', '*.txt')))
    n = min(5, len(image_paths), len(label_paths))

    '''
    fig, axs = plt.subplots(n, 1, figsize=(8, 5 * n)) 

    for idx in range(n):
        img_bgr = cv2.imread(image_paths[idx])
        # draw GT boxes and labels
        with open(label_paths[idx]) as f:
            for line in f:
                parts = line.strip().split()
                cls_name = parts[0]
                if cls_name not in CLASS_MAP:
                    continue
                cls_idx = CLASS_MAP[cls_name]
                x1, y1, x2, y2 = map(int, map(float, parts[4:8]))
                r, g, b = class_color_map[cls_idx]
                color_bgr = (b, g, r)
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color_bgr, 3)
                cv2.putText(img_bgr, cls_name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color_bgr, 2, cv2.LINE_AA)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        ax = axs[idx]
        ax.imshow(img_rgb)
        ax.axis('off')
        ax.set_title(f"Image {idx}")

    plt.tight_layout()
    plt.savefig('./output/3_1.png')
    '''
    # anchors for two scales
    anchors_big   = [(116, 90), (156, 198), (373, 326)]
    anchors_small = [(30, 61),  (62, 45),   (59, 119)]
    strides       = [32, 16]
    num_classes   = len(KITTI_CLASSES)
    bce_loss      = nn.BCEWithLogitsLoss(reduction='sum')  # objectness and classification loss
    mse_loss      = nn.MSELoss(reduction='sum')  # localization loss
    dataset = DetectKITTI(root_dir="../", img_size=(416, 416))

    # DataLoader for training data
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=yolo_collate_fn, pin_memory=True)

    # display dataset information
    print(f"Number of samples: {len(dataset)}")
    print(f"Batch shape: {next(iter(train_loader))[0].shape}")
    print(f"Classes ({num_classes}): {', '.join(KITTI_CLASSES)}")

    all_labels = []
    for idx in range(len(dataset)):
        _, tgt = dataset[idx]
        all_labels += tgt['labels'].tolist()

    # count how many boxes per class
    counts = Counter(all_labels)
    class_names = KITTI_CLASSES
    class_counts = [counts.get(i, 0) for i in range(len(KITTI_CLASSES))]

    # plot class distribution
    plt.figure(figsize=(8, 4))
    bars = plt.bar(class_names, class_counts, color='skyblue')
    plt.title("Training Dataset Class Distribution")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=45)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 5, str(int(height)), ha='center')
    plt.tight_layout()
    plt.show()