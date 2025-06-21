import os
import time
import math
import random
import warnings
from glob import glob
from collections import Counter
import json
import pickle

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim

import torchvision
from torchvision import transforms
import torchvision.ops as ops
from torchvision.ops import box_iou

from sklearn.metrics import average_precision_score

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Global variable settings
KITTI_CLASSES = [
    'Car', 'Van', 'Truck', 'Pedestrian',
    'Person_sitting', 'Cyclist', 'Tram', 'Misc'
]
CLASS_MAP = {c: i for i, c in enumerate(KITTI_CLASSES)}

# YOLO settings - 수정된 버전
anchors = [
    [(10, 13), (16, 30), (33, 23)],    # P3/8
    [(30, 61), (62, 45), (59, 119)],   # P4/16  
    [(116, 90), (156, 198), (373, 326)] # P5/32
]
strides = [8, 16, 32]  # 3개 스케일 모두 사용
num_classes = len(KITTI_CLASSES)

# Loss functions
bce_loss = nn.BCEWithLogitsLoss(reduction='sum')
mse_loss = nn.MSELoss(reduction='sum')

# Hyperparameters
batch_size = 8  # 메모리 절약을 위해 감소
num_workers = 2
num_epochs = 50  # 디버깅을 위해 에포크 수 감소
conf_threshold = 0.3

class DetectKITTI(Dataset):
    def __init__(self, root_dir, img_size=(416, 416), flip_prob=0.5):
        self.img_size = img_size
        self.flip_prob = flip_prob
        base = root_dir
        
        # 경로 수정 - 실제 KITTI 데이터셋 구조에 맞게
        image_path = os.path.join(base, 'data_object_image_2/training/image_2')
        label_path = os.path.join(base, 'data_object_label_2/training/label_2')
        
        # 경로가 존재하지 않는 경우 대체 경로 시도
        if not os.path.exists(image_path):
            image_path = os.path.join(base, 'training/image_2')
        if not os.path.exists(label_path):
            label_path = os.path.join(base, 'training/label_2')
            
        self.images = sorted(glob(os.path.join(image_path, '*.png')))
        self.labels = sorted(glob(os.path.join(label_path, '*.txt')))
        
        print(f"Found {len(self.images)} images and {len(self.labels)} labels")
        
        if len(self.images) == 0:
            raise FileNotFoundError(f"No images found in {image_path}")
        if len(self.labels) == 0:
            raise FileNotFoundError(f"No labels found in {label_path}")
            
        assert len(self.images) == len(self.labels), f"Mismatch: {len(self.images)} images vs {len(self.labels)} labels"

        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.images[idx]).convert('RGB')
            orig_w, orig_h = img.size

            boxes, labels = [], []
            
            # 라벨 파일이 존재하는지 확인
            if os.path.exists(self.labels[idx]):
                with open(self.labels[idx], 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        toks = line.split()
                        if len(toks) < 8:
                            continue
                        cls = toks[0]
                        if cls not in CLASS_MAP:
                            continue
                        
                        # KITTI 형식: class truncated occluded alpha x1 y1 x2 y2 ...
                        try:
                            x1, y1, x2, y2 = map(float, toks[4:8])
                            # 유효한 박스인지 확인
                            if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0:
                                boxes.append([x1, y1, x2, y2])
                                labels.append(CLASS_MAP[cls])
                        except (ValueError, IndexError):
                            continue

            boxes = np.array(boxes, dtype=np.float32)
            if boxes.size == 0:
                boxes = np.zeros((0, 4), dtype=np.float32)
                labels = []

            # Horizontal flip augmentation
            if random.random() < self.flip_prob:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if boxes.shape[0] > 0:
                    x1 = orig_w - boxes[:, 2]
                    x2 = orig_w - boxes[:, 0]
                    boxes[:, 0], boxes[:, 2] = x1, x2

            img_t = self.transform(img)

            # Scale boxes to new image size
            new_w, new_h = self.img_size
            sx, sy = new_w / orig_w, new_h / orig_h
            if boxes.shape[0] > 0:
                boxes[:, [0, 2]] *= sx
                boxes[:, [1, 3]] *= sy

            return img_t, {
                'boxes': torch.from_numpy(boxes),
                'labels': torch.tensor(labels, dtype=torch.int64),
                'orig_size': (orig_w, orig_h),
                'image_idx': idx
            }
        except Exception as e:
            print(f"Error loading image {idx}: {e}")
            # 빈 이미지와 타겟 반환
            empty_img = torch.zeros(3, self.img_size[1], self.img_size[0])
            return empty_img, {
                'boxes': torch.empty(0, 4),
                'labels': torch.tensor([], dtype=torch.int64),
                'orig_size': (416, 416),
                'image_idx': idx
            }

class BiasedSampler:
    """Biased sampling class"""
    
    def __init__(self, dataset, bias_strategy='dominant_class', bias_ratio=0.8):
        """
        Args:
            dataset: KITTI dataset
            bias_strategy: 'dominant_class', 'exclude_minority', 'imbalanced'
            bias_ratio: bias degree (0.5~1.0)
        """
        self.dataset = dataset
        self.bias_strategy = bias_strategy
        self.bias_ratio = bias_ratio
        
        # Collect class information for each image
        self.image_class_info = self._collect_class_info()
        
    def _collect_class_info(self):
        """Collect class information contained in each image"""
        image_class_info = {}
        
        for idx in range(len(self.dataset)):
            _, target = self.dataset[idx]
            labels = target['labels'].tolist()
            image_class_info[idx] = {
                'labels': labels,
                'dominant_class': max(set(labels), key=labels.count) if labels else -1,
                'class_counts': Counter(labels)
            }
            
        return image_class_info
    
    def create_biased_indices(self, target_size=None):
        """Generate biased sample indices"""
        if target_size is None:
            target_size = len(self.dataset)
            
        if self.bias_strategy == 'dominant_class':
            return self._dominant_class_bias(target_size)
        elif self.bias_strategy == 'exclude_minority':
            return self._exclude_minority_bias(target_size)
        elif self.bias_strategy == 'imbalanced':
            return self._imbalanced_bias(target_size)
        else:
            raise ValueError(f"Unknown bias strategy: {self.bias_strategy}")
    
    def _dominant_class_bias(self, target_size):
        """Biased sampling towards main class (Car)"""
        car_class_idx = CLASS_MAP['Car']
        
        # Images containing Car
        car_images = [idx for idx, info in self.image_class_info.items() 
                      if car_class_idx in info['labels']]
        
        # Images without Car
        no_car_images = [idx for idx, info in self.image_class_info.items() 
                         if car_class_idx not in info['labels']]
        
        # Biased sampling
        car_sample_size = int(target_size * self.bias_ratio)
        no_car_sample_size = target_size - car_sample_size
        
        selected_car = random.sample(car_images, min(car_sample_size, len(car_images)))
        selected_no_car = random.sample(no_car_images, min(no_car_sample_size, len(no_car_images)))
        
        return selected_car + selected_no_car
    
    def _exclude_minority_bias(self, target_size):
        """Exclude minority classes"""
        # Calculate overall class distribution
        all_labels = []
        for info in self.image_class_info.values():
            all_labels.extend(info['labels'])
        
        class_counts = Counter(all_labels)
        total_samples = sum(class_counts.values())
        
        # Define minority classes (less than 5% of total)
        minority_classes = [cls for cls, count in class_counts.items() 
                           if count / total_samples < 0.05]
        
        # Select only images without minority classes
        valid_images = [idx for idx, info in self.image_class_info.items()
                       if not any(cls in minority_classes for cls in info['labels'])]
        
        return random.sample(valid_images, min(target_size, len(valid_images)))
    
    def _imbalanced_bias(self, target_size):
        """Extremely imbalanced sampling"""
        # Collect images for each class
        class_images = {cls: [] for cls in range(len(KITTI_CLASSES))}
        
        for idx, info in self.image_class_info.items():
            for cls in info['labels']:
                class_images[cls].append(idx)
        
        # Biased sampling (Car: 70%, Van: 15%, others: 15%)
        selected_indices = []
        
        car_size = int(target_size * 0.7)
        van_size = int(target_size * 0.15)
        other_size = target_size - car_size - van_size
        
        # Car sampling
        car_indices = class_images[CLASS_MAP['Car']]
        selected_indices.extend(random.sample(car_indices, min(car_size, len(car_indices))))
        
        # Van sampling
        van_indices = class_images[CLASS_MAP['Van']]
        selected_indices.extend(random.sample(van_indices, min(van_size, len(van_indices))))
        
        # Other classes sampling
        other_indices = []
        for cls in range(len(KITTI_CLASSES)):
            if cls not in [CLASS_MAP['Car'], CLASS_MAP['Van']]:
                other_indices.extend(class_images[cls])
        
        other_indices = list(set(other_indices))  # Remove duplicates
        selected_indices.extend(random.sample(other_indices, min(other_size, len(other_indices))))
        
        return list(set(selected_indices))  # Remove duplicates

class BalancedSampler:
    """Balanced sampling class"""
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.image_class_info = self._collect_class_info()
    
    def _collect_class_info(self):
        """Collect class information contained in each image"""
        image_class_info = {}
        
        for idx in range(len(self.dataset)):
            _, target = self.dataset[idx]
            labels = target['labels'].tolist()
            image_class_info[idx] = {
                'labels': labels,
                'class_counts': Counter(labels)
            }
            
        return image_class_info
    
    def create_balanced_indices(self, target_size=None):
        """Generate balanced sample indices"""
        if target_size is None:
            target_size = len(self.dataset)
        
        # Collect images for each class
        class_images = {cls: [] for cls in range(len(KITTI_CLASSES))}
        
        for idx, info in self.image_class_info.items():
            for cls in info['labels']:
                class_images[cls].append(idx)
        
        # Sample equal number from each class
        samples_per_class = target_size // len(KITTI_CLASSES)
        selected_indices = []
        
        for cls in range(len(KITTI_CLASSES)):
            cls_indices = list(set(class_images[cls]))  # Remove duplicates
            if len(cls_indices) >= samples_per_class:
                selected_indices.extend(random.sample(cls_indices, samples_per_class))
            else:
                # Allow duplicates if insufficient class samples
                selected_indices.extend(cls_indices * (samples_per_class // len(cls_indices) + 1))
                selected_indices = selected_indices[:len(selected_indices) - 
                                                  (len(selected_indices) - samples_per_class)]
        
        return list(set(selected_indices))[:target_size]

class RandomSampler:
    """Random sampling class"""

    def __init__(self, dataset):
        self.dataset = dataset
        self.dataset_size = len(dataset)

    def create_random_indices(self, target_size=1500):
        """Generate random sample indices"""
        if target_size > self.dataset_size:
            target_size = self.dataset_size  # 제한: 데이터셋 크기 초과 불가

        selected_indices = random.sample(range(self.dataset_size), target_size)
        return selected_indices
    
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k, s, p):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class YOLOv4Tiny(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.num_classes = num_classes

        # Backbone - Fixed channel dimensions
        self.layer1 = nn.Sequential(
            ConvBlock(3, 16, 3, 1, 1), 
            nn.MaxPool2d(2, 2),
            ConvBlock(16, 32, 3, 1, 1), 
            nn.MaxPool2d(2, 2),
            ConvBlock(32, 64, 3, 1, 1), 
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 128, 3, 1, 1)  # Output: 128 channels
        )
        
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ConvBlock(128, 256, 3, 1, 1)  # Output: 256 channels
        )
        
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ConvBlock(256, 512, 3, 1, 1)  # Output: 512 channels
        )

        # FPN - Fixed channel dimensions
        self.lateral3 = ConvBlock(512, 256, 1, 1, 0)  # 512 -> 256
        self.lateral2 = ConvBlock(256, 256, 1, 1, 0)  # 256 -> 256
        self.lateral1 = ConvBlock(128, 256, 1, 1, 0)  # 128 -> 256

        self.smooth2 = ConvBlock(256, 256, 3, 1, 1)
        self.smooth1 = ConvBlock(256, 256, 3, 1, 1)

        # Prediction heads
        self.pred3 = nn.Conv2d(256, (num_classes + 5) * 3, 1, 1, 0)
        self.pred2 = nn.Conv2d(256, (num_classes + 5) * 3, 1, 1, 0)
        self.pred1 = nn.Conv2d(256, (num_classes + 5) * 3, 1, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        c1 = self.layer1(x)  # [B, 128, H/8, W/8]
        c2 = self.layer2(c1)  # [B, 256, H/16, W/16]
        c3 = self.layer3(c2)  # [B, 512, H/32, W/32]

        p3 = self.lateral3(c3)  # [B, 256, H/32, W/32]
        p2 = self.lateral2(c2) + self.upsample(p3)  # [B, 256, H/16, W/16]
        p1 = self.lateral1(c1) + self.upsample(p2)  # [B, 256, H/8, W/8]

        p2 = self.smooth2(p2)
        p1 = self.smooth1(p1)

        out3 = self.pred3(p3)
        out2 = self.pred2(p2)
        out1 = self.pred1(p1)

        return out1, out2, out3
    
class YOLOv4Tiny(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.num_classes = num_classes

        # Backbone - 채널 수 수정
        self.layer1 = nn.Sequential(
            ConvBlock(3, 32, 3, 1, 1), 
            nn.MaxPool2d(2, 2),
            ConvBlock(32, 64, 3, 1, 1), 
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 128, 3, 1, 1), 
            nn.MaxPool2d(2, 2),
            ConvBlock(128, 256, 3, 1, 1)  # Output: 256 channels
        )
        
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ConvBlock(256, 512, 3, 1, 1)  # Output: 512 channels
        )
        
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ConvBlock(512, 1024, 3, 1, 1)  # Output: 1024 channels
        )

        # FPN
        self.lateral3 = ConvBlock(1024, 256, 1, 1, 0)  # 1024 -> 256
        self.lateral2 = ConvBlock(512, 256, 1, 1, 0)   # 512 -> 256
        self.lateral1 = ConvBlock(256, 256, 1, 1, 0)   # 256 -> 256

        self.smooth2 = ConvBlock(256, 256, 3, 1, 1)
        self.smooth1 = ConvBlock(256, 256, 3, 1, 1)

        # Prediction heads - 3개 앵커 * (클래스 + 5)
        anchors_per_scale = 3
        output_channels = anchors_per_scale * (num_classes + 5)
        
        self.pred3 = nn.Conv2d(256, output_channels, 1, 1, 0)  # Large objects
        self.pred2 = nn.Conv2d(256, output_channels, 1, 1, 0)  # Medium objects  
        self.pred1 = nn.Conv2d(256, output_channels, 1, 1, 0)  # Small objects

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # 가중치 초기화
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Feature extraction
        c1 = self.layer1(x)  # [B, 256, H/8, W/8]
        c2 = self.layer2(c1)  # [B, 512, H/16, W/16]
        c3 = self.layer3(c2)  # [B, 1024, H/32, W/32]

        # FPN
        p3 = self.lateral3(c3)  # [B, 256, H/32, W/32]
        p2 = self.lateral2(c2) + self.upsample(p3)  # [B, 256, H/16, W/16]
        p1 = self.lateral1(c1) + self.upsample(p2)  # [B, 256, H/8, W/8]

        p2 = self.smooth2(p2)
        p1 = self.smooth1(p1)

        # Predictions
        out1 = self.pred1(p1)  # Small objects
        out2 = self.pred2(p2)  # Medium objects
        out3 = self.pred3(p3)  # Large objects

        return out1, out2, out3

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k, s, p):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

def yolo_loss(outputs, targets, device):
    """수정된 YOLO 손실 함수"""
    if not outputs or len(outputs) != 3:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    out1, out2, out3 = outputs
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    for i, (pred, stride, anchor_set) in enumerate(zip([out1, out2, out3], strides, anchors)):
        B, _, H, W = pred.shape
        
        # Reshape prediction: [B, 3*(5+num_classes), H, W] -> [B, 3, H, W, 5+num_classes]
        pred = pred.view(B, 3, 5 + num_classes, H, W).permute(0, 1, 3, 4, 2).contiguous()
        
        # Target tensor 초기화
        target_tensor = torch.zeros_like(pred, device=device)
        
        # 각 배치에 대해 타겟 생성
        for b in range(B):
            if b >= len(targets):
                continue
                
            boxes = targets[b]['boxes']
            labels = targets[b]['labels']
            
            if len(boxes) == 0:
                continue
                
            for box, cls in zip(boxes, labels):
                if len(box) != 4:
                    continue
                    
                x1, y1, x2, y2 = box.float()
                
                # Center coordinates and dimensions
                cx = (x1 + x2) / 2 / stride
                cy = (y1 + y2) / 2 / stride
                gw = (x2 - x1) / stride
                gh = (y2 - y1) / stride
                
                # Grid cell coordinates
                gi = int(torch.clamp(cx, 0, W - 1))
                gj = int(torch.clamp(cy, 0, H - 1))
                
                # 가장 적합한 앵커 찾기
                anchor_ious = []
                for aw, ah in anchor_set:
                    # IoU 계산을 위한 박스 생성
                    anchor_area = aw * ah
                    gt_area = gw * gh
                    intersection = min(aw, gw) * min(ah, gh)
                    union = anchor_area + gt_area - intersection
                    iou = intersection / (union + 1e-16)
                    anchor_ious.append(iou)
                
                best_anchor = anchor_ious.index(max(anchor_ious))
                
                # 타겟 할당
                target_tensor[b, best_anchor, gj, gi, 0] = cx - gi  # x offset
                target_tensor[b, best_anchor, gj, gi, 1] = cy - gj  # y offset
                target_tensor[b, best_anchor, gj, gi, 2] = torch.log(gw / anchor_set[best_anchor][0] + 1e-16)  # w
                target_tensor[b, best_anchor, gj, gi, 3] = torch.log(gh / anchor_set[best_anchor][1] + 1e-16)  # h
                target_tensor[b, best_anchor, gj, gi, 4] = 1.0  # objectness
                
                # 클래스 라벨
                if 0 <= cls < num_classes:
                    target_tensor[b, best_anchor, gj, gi, 5 + cls] = 1.0
        
        # 손실 계산
        pred_xy = torch.sigmoid(pred[..., 0:2])
        pred_wh = pred[..., 2:4]
        pred_conf = torch.sigmoid(pred[..., 4])
        pred_cls = torch.sigmoid(pred[..., 5:])
        
        target_xy = target_tensor[..., 0:2]
        target_wh = target_tensor[..., 2:4]
        target_conf = target_tensor[..., 4]
        target_cls = target_tensor[..., 5:]
        
        # 객체가 있는 위치만 선택
        obj_mask = target_conf == 1
        
        if obj_mask.sum() > 0:
            # Coordinate loss
            xy_loss = F.mse_loss(pred_xy[obj_mask], target_xy[obj_mask], reduction='sum')
            wh_loss = F.mse_loss(pred_wh[obj_mask], target_wh[obj_mask], reduction='sum')
            
            # Classification loss
            cls_loss = F.binary_cross_entropy(pred_cls[obj_mask], target_cls[obj_mask], reduction='sum')
            
            total_loss = total_loss + xy_loss + wh_loss + cls_loss
        
        # Confidence loss (모든 위치)
        conf_loss = F.binary_cross_entropy(pred_conf, target_conf, reduction='sum')
        total_loss = total_loss + conf_loss
    
    return total_loss / B if B > 0 else total_loss

def yolo_collate_fn(batch):
    """배치 콜레이트 함수"""
    imgs, targets = zip(*batch)
    
    # 유효한 데이터만 필터링
    valid_items = []
    for img, target in zip(imgs, targets):
        if img is not None and target is not None:
            valid_items.append((img, target))
    
    if not valid_items:
        # 빈 배치 처리
        empty_img = torch.zeros(1, 3, 416, 416)
        empty_target = [{
            'boxes': torch.empty(0, 4),
            'labels': torch.tensor([], dtype=torch.int64),
            'orig_size': (416, 416),
            'image_idx': 0
        }]
        return empty_img, empty_target
    
    imgs, targets = zip(*valid_items)
    return torch.stack(imgs), list(targets)

def decode_yolo_outputs(out1, out2, out3, img_shape, conf_threshold=0.3):
    """YOLO 출력 디코딩 함수 - 수정된 버전"""
    batch_size = out1.shape[0]
    all_predictions = []
    
    outputs = [out1, out2, out3]
    
    for batch_idx in range(batch_size):
        batch_boxes = []
        
        for i, (output, stride, anchor_set) in enumerate(zip(outputs, strides, anchors)):
            _, channels, height, width = output.shape
            
            # Reshape: [3*(5+num_classes), H, W] -> [3, H, W, 5+num_classes]
            pred = output[batch_idx].view(3, 5 + num_classes, height, width).permute(0, 2, 3, 1).contiguous()
            
            # 그리드 좌표 생성
            device = output.device
            grid_y, grid_x = torch.meshgrid(
                torch.arange(height, device=device), 
                torch.arange(width, device=device), 
                indexing='ij'
            )
            
            for anchor_idx in range(3):
                anchor_w, anchor_h = anchor_set[anchor_idx]
                
                # 예측값 추출
                pred_anchor = pred[anchor_idx]  # [H, W, 5+num_classes]
                
                # 좌표 변환
                x = (torch.sigmoid(pred_anchor[..., 0]) + grid_x) * stride
                y = (torch.sigmoid(pred_anchor[..., 1]) + grid_y) * stride
                w = torch.exp(pred_anchor[..., 2]) * anchor_w
                h = torch.exp(pred_anchor[..., 3]) * anchor_h
                
                # 신뢰도와 클래스 확률
                conf = torch.sigmoid(pred_anchor[..., 4])
                cls_probs = torch.sigmoid(pred_anchor[..., 5:])
                
                # 클래스별 최대 확률과 인덱스
                cls_conf, cls_idx = torch.max(cls_probs, dim=-1)
                final_conf = conf * cls_conf
                
                # 신뢰도 임계값 필터링
                mask = final_conf > conf_threshold
                if not mask.any():
                    continue
                
                # 유효한 예측값 추출
                valid_x = x[mask]
                valid_y = y[mask]
                valid_w = w[mask]
                valid_h = h[mask]
                valid_conf = final_conf[mask]
                valid_cls = cls_idx[mask]
                
                # 박스 좌표 계산 (중심점 -> 모서리점)
                x1 = valid_x - valid_w / 2
                y1 = valid_y - valid_h / 2
                x2 = valid_x + valid_w / 2
                y2 = valid_y + valid_h / 2
                
                # 박스를 리스트에 추가
                for j in range(len(valid_x)):
                    batch_boxes.append([
                        x1[j].item(), y1[j].item(),
                        x2[j].item(), y2[j].item(),
                        valid_conf[j].item(),
                        valid_cls[j].item()
                    ])
        
        # 텐서로 변환
        if batch_boxes:
            all_predictions.append(torch.tensor(batch_boxes, device=output.device))
        else:
            all_predictions.append(torch.empty(0, 6, device=output.device))
    
    return all_predictions

def bbox_iou(box1, box2):
    """IoU 계산 함수"""
    # box1, box2: [x1, y1, x2, y2] 형식
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    inter_area = max(inter_x2 - inter_x1, 0) * max(inter_y2 - inter_y1, 0)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
    return iou

def non_max_suppression(prediction, conf_thresh=0.3, iou_thresh=0.5):
    """Non-Maximum Suppression"""
    if isinstance(prediction, list):
        if len(prediction) == 0:
            return []
        prediction = torch.stack(prediction) if len(prediction) > 0 else torch.empty(0, 6)
    
    if prediction.numel() == 0:
        return []
    
    # 신뢰도 필터링
    mask = prediction[:, 4] >= conf_thresh
    prediction = prediction[mask]
    
    if len(prediction) == 0:
        return []

    output_boxes = []
    unique_classes = prediction[:, 5].unique()
    
    for cls in unique_classes:
        cls_mask = prediction[:, 5] == cls
        cls_boxes = prediction[cls_mask]
        
        # 신뢰도 기준 정렬
        _, sort_idx = torch.sort(cls_boxes[:, 4], descending=True)
        cls_boxes = cls_boxes[sort_idx]
        
        keep_boxes = []
        while len(cls_boxes) > 0:
            current_box = cls_boxes[0]
            keep_boxes.append(current_box)
            
            if len(cls_boxes) == 1:
                break
            
            # IoU 계산 및 필터링
            ious = []
            for i in range(1, len(cls_boxes)):
                iou = bbox_iou(
                    current_box[:4].cpu().numpy(), 
                    cls_boxes[i][:4].cpu().numpy()
                )
                ious.append(iou)
            
            if ious:
                ious = torch.tensor(ious)
                keep_mask = ious < iou_thresh
                cls_boxes = cls_boxes[1:][keep_mask]
            else:
                break
        
        output_boxes.extend(keep_boxes)
    
    return output_boxes

def train_model(model, train_loader, device, num_epochs, model_name):
    """모델 훈련 함수"""
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    model.train()
    training_history = []
    
    print(f"\n========== {model_name} Training Started ==========")
    
    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        num_batches = 0
        start_time = time.time()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        
        for batch_idx, (imgs, targets) in enumerate(progress_bar):
            try:
                imgs = imgs.to(device)
                
                # 타겟을 device로 이동
                for t in targets:
                    if 'boxes' in t:
                        t['boxes'] = t['boxes'].to(device)
                    if 'labels' in t:
                        t['labels'] = t['labels'].to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(imgs)
                
                # Loss 계산
                loss = yolo_loss(outputs, targets, device)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid loss at epoch {epoch}, batch {batch_idx}")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                running_loss += loss.item()
                num_batches += 1
                
                # 진행률 표시
                progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        scheduler.step()
        
        avg_loss = running_loss / num_batches if num_batches > 0 else 0
        epoch_time = time.time() - start_time
        
        training_history.append({
            'epoch': epoch,
            'loss': avg_loss,
            'time': epoch_time
        })
        
        print(f"[{model_name}] Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f}, Time: {epoch_time:.1f}s")
    
    return training_history

def analyze_dataset_distribution(dataset, indices, title):
    """Dataset distribution analysis"""
    all_labels = []
    
    for idx in indices:
        _, target = dataset[idx]
        all_labels.extend(target['labels'].tolist())
    
    class_counts = Counter(all_labels)
    
    print(f"\n{title} Dataset Distribution:")
    print("-" * 40)
    for i, class_name in enumerate(KITTI_CLASSES):
        count = class_counts.get(i, 0)
        percentage = (count / len(all_labels)) * 100 if all_labels else 0
        print(f"{class_name}: {count} ({percentage:.1f}%)")
    
    return class_counts

def visualize_distributions(biased_counts, balanced_counts):
    """Visualize data distributions"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Biased data distribution
    biased_values = [biased_counts.get(i, 0) for i in range(len(KITTI_CLASSES))]
    ax1.bar(KITTI_CLASSES, biased_values, color='red', alpha=0.7)
    ax1.set_title('Biased Data Distribution')
    ax1.set_ylabel('Number of Samples')
    ax1.tick_params(axis='x', rotation=45)
    
    # Balanced data distribution
    balanced_values = [balanced_counts.get(i, 0) for i in range(len(KITTI_CLASSES))]
    ax2.bar(KITTI_CLASSES, balanced_values, color='blue', alpha=0.7)
    ax2.set_title('Balanced Data Distribution')
    ax2.set_ylabel('Number of Samples')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('./output/data_distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_training_history(biased_history, balanced_history):
    """Compare training histories"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss comparison
    biased_losses = [h['loss'] for h in biased_history]
    balanced_losses = [h['loss'] for h in balanced_history]
    epochs = range(1, len(biased_losses) + 1)
    
    ax1.plot(epochs, biased_losses, 'r-', label='Biased Data', linewidth=2)
    ax1.plot(epochs, balanced_losses, 'b-', label='Balanced Data', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Training time comparison
    biased_times = [h['time'] for h in biased_history]
    balanced_times = [h['time'] for h in balanced_history]
    
    ax2.plot(epochs, biased_times, 'r-', label='Biased Data', linewidth=2)
    ax2.plot(epochs, balanced_times, 'b-', label='Balanced Data', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Training Time per Epoch Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./output/training_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_experiment_results(biased_history, balanced_history, biased_counts, balanced_counts):
    """Save experiment results"""
    results = {
        'biased_training_history': biased_history,
        'balanced_training_history': balanced_history,
        'biased_data_distribution': dict(biased_counts),
        'balanced_data_distribution': dict(balanced_counts),
        'experiment_config': {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'img_size': (416, 416),
            'bias_strategy': 'dominant_class',
            'bias_ratio': 0.8
        }
    }
    
    # Save as JSON
    with open('./output/experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save as pickle too (for later model loading)
    with open('./output/experiment_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("Experiment results saved to './output/' folder.")

def calculate_gini_coefficient(counts):
    """Calculate Gini coefficient to measure imbalance (0: perfectly balanced, 1: completely imbalanced)"""
    values = list(counts.values())
    if sum(values) == 0:
        return 0
    
    values.sort()
    n = len(values)
    cumsum = np.cumsum(values)
    return (n + 1 - 2 * sum((n + 1 - i) * y for i, y in enumerate(values))) / (n * sum(values))

def yolo_to_kitti_format(pred_boxes, class_names, confs, img_id):
    lines = []
    for i in range(len(pred_boxes)):
        cls_name = class_names[i]
        x1, y1, x2, y2 = pred_boxes[i]
        score = confs[i]
        line = f"{cls_name} -1 -1 -1 {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} -1 -1 -1 -1 -1 -1 -1 {score:.2f}"
        lines.append(line)
    with open(f"./devkit_results/{img_id}.txt", 'w') as f:
        f.write('\n'.join(lines))

import numpy as np
from collections import defaultdict

def bbox_iou(box1, box2):
    """ IoU 계산 함수 (x1, y1, x2, y2) 좌표형식 """
    # box1, box2 : (4,) numpy arrays or tensors
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    inter_area = max(inter_x2 - inter_x1, 0) * max(inter_y2 - inter_y1, 0)
    box1_area = (box1[2]-box1[0]) * (box1[3]-box1[1])
    box2_area = (box2[2]-box2[0]) * (box2[3]-box2[1])
    iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
    return iou

def non_max_suppression(prediction, conf_thresh=0.5, iou_thresh=0.5):
    """NMS 수정 버전"""
    if isinstance(prediction, list):
        prediction = torch.stack(prediction) if prediction else torch.empty(0, 6)
    
    if prediction.numel() == 0:
        return []
    
    # Confidence threshold 필터링
    mask = prediction[:, 4] >= conf_thresh
    prediction = prediction[mask]
    
    if prediction.numel() == 0:
        return []

    # 클래스별 NMS
    output_boxes = []
    unique_classes = prediction[:, 5].unique()
    
    for cls in unique_classes:
        cls_mask = prediction[:, 5] == cls
        cls_boxes = prediction[cls_mask]
        
        # Confidence 기준 정렬
        _, sort_idx = torch.sort(cls_boxes[:, 4], descending=True)
        cls_boxes = cls_boxes[sort_idx]
        
        keep_boxes = []
        while cls_boxes.size(0) > 0:
            # 가장 높은 confidence의 박스 선택
            current_box = cls_boxes[0]
            keep_boxes.append(current_box)
            
            if cls_boxes.size(0) == 1:
                break
            
            # 나머지 박스들과 IoU 계산
            ious = []
            for i in range(1, cls_boxes.size(0)):
                iou = bbox_iou(current_box[:4].cpu().numpy(), cls_boxes[i][:4].cpu().numpy())
                ious.append(iou)
            
            ious = torch.tensor(ious)
            # IoU가 임계값보다 낮은 박스들만 유지
            keep_mask = ious < iou_thresh
            cls_boxes = cls_boxes[1:][keep_mask]
        
        output_boxes.extend(keep_boxes)
    
    return output_boxes

def decode_yolo_outputs(out1, out2, out3, img_shape, conf_threshold=0.3):
    """YOLO 출력을 bbox 형식으로 디코딩 - 수정된 버전"""
    predictions = []
    outputs = [out1, out2, out3]
    stride_list = [8, 16, 32]
    anchor_list = [
        [(10, 13), (16, 30), (33, 23)],    # P3/8
        [(30, 61), (62, 45), (59, 119)],   # P4/16  
        [(116, 90), (156, 198), (373, 326)] # P5/32
    ]
    
    batch_size = out1.shape[0]
    all_predictions = []
    
    for batch_idx in range(batch_size):
        batch_boxes = []
        
        for i, (output, stride, anchors) in enumerate(zip(outputs, stride_list, anchor_list)):
            # output shape: [batch, (5+num_classes)*3, H, W]
            _, channels, height, width = output.shape
            num_attrs = 5 + num_classes  # x, y, w, h, conf + classes
            
            # Reshape: [batch, 3, num_attrs, H, W] -> [batch, 3, H, W, num_attrs]
            pred = output[batch_idx].view(3, num_attrs, height, width).permute(0, 2, 3, 1).contiguous()
            
            # Grid coordinates
            grid_y, grid_x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
            grid_x = grid_x.to(output.device).float()
            grid_y = grid_y.to(output.device).float()
            
            for anchor_idx in range(3):
                anchor_w, anchor_h = anchors[anchor_idx]
                
                # Get predictions for this anchor
                pred_anchor = pred[anchor_idx]  # [H, W, num_attrs]
                
                # Extract components
                x = pred_anchor[..., 0]
                y = pred_anchor[..., 1] 
                w = pred_anchor[..., 2]
                h = pred_anchor[..., 3]
                conf = pred_anchor[..., 4]
                cls_pred = pred_anchor[..., 5:]
                
                # Apply sigmoid and exponential
                x = torch.sigmoid(x) + grid_x
                y = torch.sigmoid(y) + grid_y
                w = torch.exp(w) * anchor_w
                h = torch.exp(h) * anchor_h
                conf = torch.sigmoid(conf)
                cls_scores = torch.sigmoid(cls_pred)
                
                # Scale to image coordinates
                x *= stride
                y *= stride
                
                # Convert to corner coordinates
                x1 = x - w / 2
                y1 = y - h / 2
                x2 = x + w / 2
                y2 = y + h / 2
                
                # Get class predictions
                cls_conf, cls_idx = torch.max(cls_scores, dim=-1)
                final_conf = conf * cls_conf
                
                # Filter by confidence
                valid_mask = final_conf > conf_threshold
                if not valid_mask.any():
                    continue
                
                # Extract valid predictions
                valid_x1 = x1[valid_mask]
                valid_y1 = y1[valid_mask]
                valid_x2 = x2[valid_mask]
                valid_y2 = y2[valid_mask]
                valid_conf = final_conf[valid_mask]
                valid_cls = cls_idx[valid_mask]
                
                # Collect boxes
                for j in range(len(valid_x1)):
                    batch_boxes.append([
                        valid_x1[j].item(),
                        valid_y1[j].item(), 
                        valid_x2[j].item(),
                        valid_y2[j].item(),
                        valid_conf[j].item(),
                        valid_cls[j].item()
                    ])
        
        all_predictions.append(torch.tensor(batch_boxes) if batch_boxes else torch.empty(0, 6))
    
    return all_predictions

def evaluate_model_debug(model, dataloader, device, iou_threshold=0.5, conf_threshold=0.3):
    """디버깅용 평가 함수"""
    model.eval()
    all_detections = []
    all_annotations = []
    
    total_predictions = 0
    total_gt = 0
    valid_images = 0

    with torch.no_grad():
        for batch_idx, (imgs, targets) in enumerate(dataloader):
            imgs = imgs.to(device)
            
            # 모델 추론
            try:
                out1, out2, out3 = model(imgs)
                print(f"Batch {batch_idx}: Output shapes - out1: {out1.shape}, out2: {out2.shape}, out3: {out3.shape}")
            except Exception as e:
                print(f"Model inference error: {e}")
                continue
            
            # YOLO 출력 디코딩
            try:
                predictions = decode_yolo_outputs(out1, out2, out3, imgs.shape[2:], conf_threshold)
                print(f"Batch {batch_idx}: Decoded {len(predictions)} predictions")
                
                # 배치의 각 이미지에 대해 처리
                for i, pred in enumerate(predictions):
                    print(f"  Image {i}: {len(pred)} raw detections")
                    
                    # NMS 적용
                    if len(pred) > 0:
                        nms_boxes = non_max_suppression_debug(pred, conf_threshold, iou_threshold)
                        print(f"  Image {i}: {len(nms_boxes)} after NMS")
                        total_predictions += len(nms_boxes)
                    else:
                        nms_boxes = []
                    
                    all_detections.append(nms_boxes)
                    
            except Exception as e:
                print(f"Decoding error: {e}")
                all_detections.append([])
            
            # GT 처리
            for i, target in enumerate(targets):
                gt_boxes = []
                for box, label in zip(target['boxes'], target['labels']):
                    gt_boxes.append((box.cpu().numpy(), label.item()))
                    total_gt += 1
                
                all_annotations.append(gt_boxes)
                print(f"  GT Image {i}: {len(gt_boxes)} ground truth boxes")
                
                if len(gt_boxes) > 0:
                    valid_images += 1
            
            # 처음 몇 배치만 상세히 출력
            if batch_idx >= 5:
                break

    print(f"\nSummary:")
    print(f"Total predictions: {total_predictions}")
    print(f"Total ground truth: {total_gt}")
    print(f"Valid images with GT: {valid_images}")
    
    if total_predictions == 0:
        print("WARNING: No predictions found! Check model inference and decoding.")
        return {}, 0.0
    
    if total_gt == 0:
        print("WARNING: No ground truth found! Check dataset loading.")
        return {}, 0.0

    # mAP 계산
    try:
        average_precisions = compute_mean_average_precision(all_detections, all_annotations, iou_threshold)
        mAP = np.mean(list(average_precisions.values())) if average_precisions else 0.0
        
        print(f"\nClass-wise AP:")
        for cls, ap in average_precisions.items():
            class_name = KITTI_CLASSES[cls] if cls < len(KITTI_CLASSES) else f"Class_{cls}"
            print(f"  {class_name}: {ap:.4f}")
        
        print(f"mAP @ IoU={iou_threshold}: {mAP:.4f}")
        return average_precisions, mAP
        
    except Exception as e:
        print(f"mAP calculation error: {e}")
        return {}, 0.0

# 3. 디버깅용 NMS 함수
def non_max_suppression_debug(prediction, conf_thresh=0.3, iou_thresh=0.5):
    """디버깅용 NMS 함수"""
    if isinstance(prediction, list):
        if len(prediction) == 0:
            return []
        prediction = torch.stack(prediction) if len(prediction) > 0 else torch.empty(0, 6)
    
    if prediction.numel() == 0 or len(prediction) == 0:
        return []
    
    print(f"    NMS input: {len(prediction)} boxes")
    print(f"    Confidence range: {prediction[:, 4].min():.3f} - {prediction[:, 4].max():.3f}")
    
    # Confidence threshold 적용
    mask = prediction[:, 4] >= conf_thresh
    prediction = prediction[mask]
    print(f"    After conf filter: {len(prediction)} boxes")
    
    if len(prediction) == 0:
        return []

    output_boxes = []
    unique_classes = prediction[:, 5].unique()
    print(f"    Unique classes: {unique_classes.tolist()}")
    
    for cls in unique_classes:
        cls_mask = prediction[:, 5] == cls
        cls_boxes = prediction[cls_mask]
        print(f"    Class {int(cls)}: {len(cls_boxes)} boxes")
        
        # Confidence로 정렬
        _, sort_idx = torch.sort(cls_boxes[:, 4], descending=True)
        cls_boxes = cls_boxes[sort_idx]
        
        keep_boxes = []
        while len(cls_boxes) > 0:
            current_box = cls_boxes[0]
            keep_boxes.append(current_box)
            
            if len(cls_boxes) == 1:
                break
            
            # IoU 계산
            ious = []
            for i in range(1, len(cls_boxes)):
                iou = bbox_iou(current_box[:4].cpu().numpy(), cls_boxes[i][:4].cpu().numpy())
                ious.append(iou)
            
            if len(ious) > 0:
                ious = torch.tensor(ious)
                keep_mask = ious < iou_thresh
                cls_boxes = cls_boxes[1:][keep_mask]
            else:
                break
        
        output_boxes.extend(keep_boxes)
        print(f"    Class {int(cls)} after NMS: {len(keep_boxes)} boxes")
    
    return output_boxes

# 4. 메인 평가 코드 수정
def main_evaluation():
    """메인 평가 함수"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 데이터셋 로드
    print("Loading dataset...")
    full_dataset = DetectKITTI(root_dir="../", img_size=(416, 416))
    print(f"Total dataset size: {len(full_dataset)}")
    
    # 작은 샘플로 테스트
    random_sampler = RandomSampler(full_dataset)
    sample_size = min(50, len(full_dataset))  # 작은 샘플로 먼저 테스트
    random_indices = random_sampler.create_random_indices(sample_size)
    random_dataset = Subset(full_dataset, random_indices)
    test_loader = DataLoader(random_dataset, batch_size=1, shuffle=False, 
                           num_workers=0, collate_fn=yolo_collate_fn)  # num_workers=0으로 설정
    
    # 모델 로드 및 평가
    try:
        print("\nLoading and evaluating biased model...")
        test_biased_model = YOLOv4Tiny(num_classes=len(KITTI_CLASSES)).to(device)
        test_biased_model.load_state_dict(torch.load('../models/biased_model.pth', map_location=device))
        biased_results = evaluate_model_debug(test_biased_model, test_loader, device)
        
        print("\nLoading and evaluating balanced model...")
        test_balanced_model = YOLOv4Tiny(num_classes=len(KITTI_CLASSES)).to(device)  
        test_balanced_model.load_state_dict(torch.load('../models/balanced_model.pth', map_location=device))
        balanced_results = evaluate_model_debug(test_balanced_model, test_loader, device)
        
    except Exception as e:
        print(f"Model loading/evaluation error: {e}")
        import traceback
        traceback.print_exc()

def evaluate_model(model, dataloader, device, iou_threshold=0.5, conf_threshold=0.5):
    model.eval()
    all_detections = []
    all_annotations = []

    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = imgs.to(device)
            out1, out2, out3 = model(imgs)  # YOLO 모델의 실제 출력 형식
            
            # YOLO 출력을 bbox 형식으로 변환
            predictions = decode_yolo_outputs(out1, out2, out3, imgs.shape[2:])
            
            # NMS 적용
            pred_boxes = []
            for pred in predictions:
                if len(pred) > 0:
                    nms_boxes = non_max_suppression(pred, conf_threshold, iou_threshold)
                    pred_boxes.extend(nms_boxes)

            # GT targets 변환
            gt_boxes = []
            for t in targets:
                for box, label in zip(t['boxes'], t['labels']):
                    gt_boxes.append((box.cpu().numpy(), label.item()))
            
            all_detections.append(pred_boxes)
            all_annotations.append(gt_boxes)

    # mAP 계산
    average_precisions = compute_mean_average_precision(all_detections, all_annotations, iou_threshold)
    mAP = np.mean(list(average_precisions.values())) if average_precisions else 0.0
    
    print(f"mAP @ IoU={iou_threshold}: {mAP:.4f}")
    return average_precisions, mAP

def compute_mean_average_precision(detections, annotations, iou_threshold):
    """mAP 계산 수정 버전"""
    class_tp = defaultdict(list)
    class_fp = defaultdict(list) 
    class_scores = defaultdict(list)
    class_num_gt = defaultdict(int)

    # GT 개수 먼저 계산
    for gts in annotations:
        for gt in gts:
            class_num_gt[gt[1]] += 1

    for preds, gts in zip(detections, annotations):
        # 이미 매칭된 GT 추적
        matched_gts = set()
        
        for pred in preds:
            if isinstance(pred, torch.Tensor):
                pred_box = pred[:4].cpu().numpy()
                pred_conf = pred[4].cpu().item()
                pred_cls = int(pred[5].cpu().item())
            else:
                pred_box = pred[:4]
                pred_conf = pred[4]
                pred_cls = int(pred[5])

            # 같은 클래스의 GT만 필터링
            gt_same_class = [(i, gt) for i, gt in enumerate(gts) if gt[1] == pred_cls]
            
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, (_, gt) in enumerate(gt_same_class):
                iou = bbox_iou(pred_box, gt[0])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # TP/FP 판정
            if best_iou >= iou_threshold and best_gt_idx not in matched_gts:
                class_tp[pred_cls].append(1)
                class_fp[pred_cls].append(0)
                matched_gts.add(best_gt_idx)
            else:
                class_tp[pred_cls].append(0)
                class_fp[pred_cls].append(1)
            
            class_scores[pred_cls].append(pred_conf)

    # 클래스별 AP 계산
    average_precisions = {}
    for cls in class_tp.keys():
        if class_num_gt[cls] == 0:
            continue
            
        tp = np.array(class_tp[cls])
        fp = np.array(class_fp[cls])
        scores = np.array(class_scores[cls])

        # Confidence 기준 정렬
        indices = np.argsort(-scores)
        tp = tp[indices]
        fp = fp[indices]

        # 누적 계산
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / class_num_gt[cls]
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

        # AP 계산 (11-point interpolation)
        ap = 0
        for t in np.linspace(0, 1, 11):
            if np.any(recalls >= t):
                p = precisions[recalls >= t].max()
            else:
                p = 0
            ap += p / 11
            
        average_precisions[cls] = ap

    return average_precisions

# Global variable settings
KITTI_CLASSES = [
    'Car', 'Van', 'Truck', 'Pedestrian',
    'Person_sitting', 'Cyclist', 'Tram', 'Misc'
]
CLASS_MAP = {c: i for i, c in enumerate(KITTI_CLASSES)}

# YOLO settings
anchors_big = [(116, 90), (156, 198), (373, 326)]
anchors_small = [(30, 61), (62, 45), (59, 119)]
strides = [32, 16]
num_classes = len(KITTI_CLASSES)
bce_loss = nn.BCEWithLogitsLoss(reduction='sum')
mse_loss = nn.MSELoss(reduction='sum')

# Hyperparameters
batch_size = 16  # Reduced for memory saving
num_workers = 2
num_epochs = 200

# eval parameter
conf_threshold = 0.3
'''
if __name__ == "__main__":
    main_evaluation()
'''
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create result folders
    os.makedirs('./output', exist_ok=True)
    os.makedirs('../models', exist_ok=True)
    
    # Load original dataset
    print("Loading dataset...")
    full_dataset = DetectKITTI(root_dir="../", img_size=(416, 416))
    print(f"Total dataset size: {len(full_dataset)}")
    
    # Create sampling objects
    biased_sampler = BiasedSampler(full_dataset, bias_strategy='dominant_class', bias_ratio=0.8)
    balanced_sampler = BalancedSampler(full_dataset)
    random_sampler = RandomSampler(full_dataset)
    
    # Generate sample indices (use smaller size for experiment)
    sample_size = min(1000, len(full_dataset))  # Use only part of the full data
    
    
    print("\nBiased sampling...")
    biased_indices = biased_sampler.create_biased_indices(sample_size)
    
    print("Balanced sampling...")
    balanced_indices = balanced_sampler.create_balanced_indices(sample_size)
    
    print("Random sampling...")
    random_indices = random_sampler.create_random_indices(sample_size)
    
    # Analyze dataset distributions
    #biased_counts = analyze_dataset_distribution(full_dataset, biased_indices, "Biased")
    #balanced_counts = analyze_dataset_distribution(full_dataset, balanced_indices, "Balanced")
    #random_counts = analyze_dataset_distribution(full_dataset, random_indices, "Random")
    
    # Visualize distributions
    #visualize_distributions(biased_counts, balanced_counts, random_counts)
    
    # Create subsets
    biased_dataset = Subset(full_dataset, biased_indices)
    balanced_dataset = Subset(full_dataset, balanced_indices)
    random_dataset = Subset(full_dataset, random_indices)
    
    # Create data loaders
    biased_loader = DataLoader(biased_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, collate_fn=yolo_collate_fn, pin_memory=True)
    balanced_loader = DataLoader(balanced_dataset, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers, collate_fn=yolo_collate_fn, pin_memory=True)
    random_loader = DataLoader(random_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, collate_fn=yolo_collate_fn, pin_memory=True)
    '''
    # Train models as before
    print("\n" + "="*50)
    print("Training model with biased data")
    print("="*50)
    biased_model = YOLOv4Tiny(num_classes=len(KITTI_CLASSES)).to(device)
    biased_history = train_model(biased_model, biased_loader, device, num_epochs, "Biased Model")
    torch.save(biased_model.state_dict(), '../models/biased_model.pth')
    
    print("\n" + "="*50)
    print("Training model with balanced data")
    print("="*50)
    balanced_model = YOLOv4Tiny(num_classes=len(KITTI_CLASSES)).to(device)
    balanced_history = train_model(balanced_model, balanced_loader, device, num_epochs, "Balanced Model")
    torch.save(balanced_model.state_dict(), '../models/balanced_model.pth')
    
    # Compare training results
    compare_training_history(biased_history, balanced_history)
    save_experiment_results(biased_history, balanced_history, biased_counts, balanced_counts)
    '''
    
    # === 여기서부터 저장된 모델 불러와서 random 샘플 데이터로 평가 ===
    print("\n" + "="*50)
    print("Load saved models and evaluate on Random sampled test data")
    print("="*50)
    
    # Prepare random indices and dataset for evaluation
    random_indices = random_sampler.create_random_indices(sample_size)
    random_dataset = Subset(full_dataset, random_indices)
    test_loader = DataLoader(random_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=yolo_collate_fn, pin_memory=True)
    
    # Load models
    test_biased_model = YOLOv4Tiny(num_classes=len(KITTI_CLASSES)).to(device)
    test_biased_model.load_state_dict(torch.load('../models/biased_model.pth'))
    test_biased_model.eval()
    
    test_balanced_model = YOLOv4Tiny(num_classes=len(KITTI_CLASSES)).to(device)
    test_balanced_model.load_state_dict(torch.load('../models/balanced_model.pth'))
    test_balanced_model.eval()
    
    print("Evaluating biased model...")
    biased_results = evaluate_model(test_biased_model, test_loader, device)
    
    print("Evaluating balanced model...")
    balanced_results = evaluate_model(test_balanced_model, test_loader, device)
    
    print("\nEvaluation completed on Random sampled test dataset.")

    
