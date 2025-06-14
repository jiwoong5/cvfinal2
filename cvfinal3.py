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
plt.rcParams['font.family'] = 'DejaVu Sans'  # 기본 폰트로 변경
plt.rcParams['axes.unicode_minus'] = False

class DetectKITTI(Dataset):
    def __init__(self, root_dir, img_size=(416, 416), flip_prob=0.5):
        self.img_size = img_size
        self.flip_prob = flip_prob
        base = root_dir
        self.images = sorted(glob(os.path.join(base, 'data_object_image_2/training/image_2', '*.png')))
        self.labels = sorted(glob(os.path.join(base, 'training/label_2', '*.txt')))
        assert len(self.images) == len(self.labels)

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
        img = Image.open(self.images[idx]).convert('RGB')
        orig_w, orig_h = img.size

        boxes, labels = [], []
        with open(self.labels[idx], 'r') as f:
            for line in f:
                toks = line.split()
                cls = toks[0]
                if cls not in CLASS_MAP:
                    continue
                x1, y1, x2, y2 = map(float, toks[4:8])
                boxes.append([x1, y1, x2, y2])
                labels.append(CLASS_MAP[cls])

        boxes = np.array(boxes, dtype=np.float32)
        if boxes.size == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = []

        if random.random() < self.flip_prob:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if boxes.shape[0] > 0:
                x1 = orig_w - boxes[:, 2]
                x2 = orig_w - boxes[:, 0]
                boxes[:, 0], boxes[:, 2] = x1, x2

        img_t = self.transform(img)

        new_w, new_h = self.img_size
        sx, sy = new_w / orig_w, new_h / orig_h
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy

        return img_t, {
            'boxes': torch.from_numpy(boxes),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'orig_size': (orig_w, orig_h),
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

def yolo_loss(out1, out2, targets):
    losses = {'xy': 0, 'wh': 0, 'obj': 0, 'cls': 0}
    for i, pred in enumerate([out1, out2]):
        B, _, H, W = pred.shape
        pred = pred.view(B, 3, 5 + num_classes, H, W).permute(0, 1, 3, 4, 2)

        ttarget = torch.zeros_like(pred)

        for b in range(B):
            boxes = targets[b]['boxes']
            labels = targets[b]['labels']
            for box, cls in zip(boxes, labels):
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2 / strides[i]
                cy = (y1 + y2) / 2 / strides[i]
                gw = (x2 - x1) / strides[i]
                gh = (y2 - y1) / strides[i]

                gi = min(int(cx), W - 1)
                gj = min(int(cy), H - 1)

                anchor_list = anchors_big if i == 0 else anchors_small
                ratios = [(gw / aw, gh / ah) for aw, ah in anchor_list]
                idx = max(range(3), key=lambda a: min(ratios[a]))

                ttarget[b, idx, gj, gi, 0] = cx - gi
                ttarget[b, idx, gj, gi, 1] = cy - gj
                ttarget[b, idx, gj, gi, 2] = torch.log(gw / anchor_list[idx][0] + 1e-16)
                ttarget[b, idx, gj, gi, 3] = torch.log(gh / anchor_list[idx][1] + 1e-16)
                ttarget[b, idx, gj, gi, 4] = 1.0
                ttarget[b, idx, gj, gi, 5 + cls] = 1.0

        pxy, txy = pred[..., 0:2], ttarget[..., 0:2]
        pwh, twh = pred[..., 2:4], ttarget[..., 2:4]
        pobj, tobj = pred[..., 4], ttarget[..., 4]
        pcls, tcls = pred[..., 5:], ttarget[..., 5:]

        obj_mask = tobj == 1

        losses['xy'] += mse_loss(pxy[obj_mask], txy[obj_mask])
        losses['wh'] += mse_loss(pwh[obj_mask], twh[obj_mask])
        losses['obj'] += bce_loss(pobj, tobj)
        losses['cls'] += bce_loss(pcls[obj_mask], tcls[obj_mask])

    total_loss = losses['xy'] + losses['wh'] + losses['obj'] + losses['cls']
    return total_loss / out1.shape[0] if out1.shape[0] > 0 else torch.tensor(0.0, requires_grad=True)

def yolo_collate_fn(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs), list(targets)

def train_model(model, train_loader, device, num_epochs, model_name):
    """Model training function"""
    criterion = yolo_loss
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    model.train()
    training_history = []
    
    print(f"\n========== {model_name} Training Started ==========")
    
    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        start_time = time.time()
        
        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            imgs = imgs.to(device)
            for t in targets:
                t['boxes'] = t['boxes'].to(device)
                t['labels'] = t['labels'].to(device)
            
            optimizer.zero_grad()
            out1, out2, out3 = model(imgs)
            loss = criterion(out1, out2, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        scheduler.step()
        avg_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
        epoch_time = time.time() - start_time
        
        training_history.append({
            'epoch': epoch,
            'loss': avg_loss,
            'time': epoch_time
        })
        
        if epoch % 5 == 0:
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
num_epochs = 20

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    '''
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
    
    # Generate sample indices (use smaller size for experiment)
    sample_size = min(1000, len(full_dataset))  # Use only part of the full data
    
    print("\nBiased sampling...")
    biased_indices = biased_sampler.create_biased_indices(sample_size)
    
    print("Balanced sampling...")
    balanced_indices = balanced_sampler.create_balanced_indices(sample_size)
    
    # Analyze dataset distributions
    biased_counts = analyze_dataset_distribution(full_dataset, biased_indices, "Biased")
    balanced_counts = analyze_dataset_distribution(full_dataset, balanced_indices, "Balanced")
    
    # Visualize distributions
    visualize_distributions(biased_counts, balanced_counts)
    
    # Create subsets
    biased_dataset = Subset(full_dataset, biased_indices)
    balanced_dataset = Subset(full_dataset, balanced_indices)
    
    # Create data loaders
    biased_loader = DataLoader(biased_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, collate_fn=yolo_collate_fn, pin_memory=True)
    balanced_loader = DataLoader(balanced_dataset, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers, collate_fn=yolo_collate_fn, pin_memory=True)
    
    # Model 1: Train with biased data
    print("\n" + "="*50)
    print("Training model with biased data")
    print("="*50)
    
    biased_model = YOLOv4Tiny(num_classes=len(KITTI_CLASSES)).to(device)
    biased_history = train_model(biased_model, biased_loader, device, num_epochs, "Biased Model")
    
    # Save biased model
    torch.save(biased_model.state_dict(), '../models/biased_model.pth')
    
    # Model 2: Train with balanced data
    print("\n" + "="*50)
    print("Training model with balanced data")
    print("="*50)
    
    balanced_model = YOLOv4Tiny(num_classes=len(KITTI_CLASSES)).to(device)
    balanced_history = train_model(balanced_model, balanced_loader, device, num_epochs, "Balanced Model")
    
    # Save balanced model
    torch.save(balanced_model.state_dict(), '../models/balanced_model.pth')
    
    # Compare training results
    compare_training_history(biased_history, balanced_history)
    
    # Save experiment results
    save_experiment_results(biased_history, balanced_history, biased_counts, balanced_counts)
    
    # Final performance comparison
    print("\n" + "="*50)
    print("Experiment Results Summary")
    print("="*50)
    
    final_biased_loss = biased_history[-1]['loss']
    final_balanced_loss = balanced_history[-1]['loss']
    
    print(f"Biased model final loss: {final_biased_loss:.4f}")
    print(f"Balanced model final loss: {final_balanced_loss:.4f}")
    print(f"Loss difference: {abs(final_biased_loss - final_balanced_loss):.4f}")
    
    # Average training time comparison
    avg_biased_time = sum(h['time'] for h in biased_history) / len(biased_history)
    avg_balanced_time = sum(h['time'] for h in balanced_history) / len(balanced_history)
    
    print(f"Average biased training time per epoch: {avg_biased_time:.2f}s")
    print(f"Average balanced training time per epoch: {avg_balanced_time:.2f}s")
    
    # Data imbalance metrics
    biased_gini = calculate_gini_coefficient(biased_counts)
    balanced_gini = calculate_gini_coefficient(balanced_counts)
    
    print(f"Biased data Gini coefficient: {biased_gini:.4f}")
    print(f"Balanced data Gini coefficient: {balanced_gini:.4f}")
    
    print("\nExperiment completed successfully!")
    '''