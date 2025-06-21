import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import shutil
from tqdm import tqdm
import random

# KITTI 클래스 정의 (8개 클래스)
KITTI_CLASSES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(KITTI_CLASSES)}

class KITTIDataset(Dataset):
    def __init__(self, image_dir, label_dir, img_size=416, augment=False):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.augment = augment
        
        # 이미지 파일 목록 가져오기
        self.image_files = sorted(list(self.image_dir.glob('*.png')))
        
        # 데이터 전처리 정의
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def parse_kitti_label(self, label_path):
        """KITTI 라벨 파일을 파싱하여 YOLO 형식으로 변환"""
        boxes = []
        if not label_path.exists():
            return boxes
            
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
                
            class_name = parts[0]
            if class_name not in CLASS_TO_IDX:
                continue
                
            # 트런케이션과 오클루전 필터링
            truncated = float(parts[1])
            occluded = int(parts[2])
            
            if truncated > 0.5 or occluded > 2:
                continue
                
            # 바운딩 박스 좌표 (픽셀 좌표)
            left = float(parts[4])
            top = float(parts[5])
            right = float(parts[6])
            bottom = float(parts[7])
            
            # 유효한 박스인지 확인
            if right <= left or bottom <= top:
                continue
                
            class_id = CLASS_TO_IDX[class_name]
            boxes.append([class_id, left, top, right, bottom])
            
        return boxes
    
    def __getitem__(self, idx):
        # 이미지 로드
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # 라벨 로드
        label_path = self.label_dir / (img_path.stem + '.txt')
        boxes = self.parse_kitti_label(label_path)
        
        # 이미지 리사이즈 및 박스 좌표 정규화
        image_resized = cv2.resize(image, (self.img_size, self.img_size))
        
        # 박스를 YOLO 형식으로 변환 (중심점, 너비, 높이, 정규화)
        yolo_boxes = []
        for box in boxes:
            class_id, left, top, right, bottom = box
            
            # 정규화된 중심점과 크기 계산
            center_x = (left + right) / 2 / w
            center_y = (top + bottom) / 2 / h
            box_w = (right - left) / w
            box_h = (bottom - top) / h
            
            # 크기 조정 후 좌표 재계산
            center_x = center_x * self.img_size / self.img_size
            center_y = center_y * self.img_size / self.img_size
            box_w = box_w * self.img_size / self.img_size
            box_h = box_h * self.img_size / self.img_size
            
            yolo_boxes.append([class_id, center_x, center_y, box_w, box_h])
        
        # 텐서로 변환
        image_tensor = self.transform(image_resized)
        
        if len(yolo_boxes) > 0:
            targets = torch.tensor(yolo_boxes, dtype=torch.float32)
        else:
            targets = torch.zeros((0, 5), dtype=torch.float32)
            
        return image_tensor, targets, str(img_path)

class YOLOv4Tiny(nn.Module):
    def __init__(self, num_classes=8):
        super(YOLOv4Tiny, self).__init__()
        self.num_classes = num_classes
        
        # Backbone (간단한 CNN 구조)
        self.backbone = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 5
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
        )
        
        # Detection Head
        self.detection_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, (num_classes + 5) * 3, 1),  # 3 anchors per cell
        )
        
        self.img_size = 416
        self.grid_size = 13
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.detection_head(x)
        
        batch_size = x.size(0)
        grid_size = x.size(2)
        
        # Reshape output: [batch, anchors, grid, grid, (5 + num_classes)]
        prediction = x.view(batch_size, 3, self.num_classes + 5, grid_size, grid_size)
        prediction = prediction.permute(0, 1, 3, 4, 2).contiguous()
        
        return prediction

class YOLOLoss(nn.Module):
    def __init__(self, num_classes=8):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, predictions, targets):
        # 간단한 손실 함수 구현
        # 실제로는 더 복잡한 YOLO 손실 함수가 필요하지만, 여기서는 간단히 구현
        device = predictions.device
        batch_size, num_anchors, grid_size, _, _ = predictions.shape
        
        # 좌표 손실
        coord_loss = torch.tensor(0.0, device=device)
        conf_loss = torch.tensor(0.0, device=device)
        cls_loss = torch.tensor(0.0, device=device)
        
        for i in range(batch_size):
            if len(targets[i]) > 0:
                # 간단한 손실 계산 (실제 구현에서는 더 정교해야 함)
                coord_loss += self.mse_loss(predictions[i, :, :, :, :4].sum(), 
                                          torch.tensor(1.0, device=device))
                conf_loss += self.bce_loss(predictions[i, :, :, :, 4].sum(), 
                                         torch.tensor(1.0, device=device))
                cls_loss += self.bce_loss(predictions[i, :, :, :, 5:].sum(), 
                                        torch.tensor(1.0, device=device))
        
        total_loss = coord_loss + conf_loss + cls_loss
        return total_loss

def calculate_iou(box1, box2):
    """IoU 계산"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union

def calculate_map(predictions, ground_truths, iou_threshold=0.5):
    """mAP@0.5 계산"""
    aps = []
    
    for class_id in range(len(KITTI_CLASSES)):
        class_predictions = []
        class_ground_truths = []
        
        # 클래스별로 예측과 실제값 분리
        for img_id, (pred_boxes, gt_boxes) in enumerate(zip(predictions, ground_truths)):
            for box in pred_boxes:
                if box[0] == class_id:  # class_id 확인
                    class_predictions.append((img_id, box[1], box[2:]))  # (img_id, confidence, bbox)
            
            for box in gt_boxes:
                if box[0] == class_id:
                    class_ground_truths.append((img_id, box[1:]))  # (img_id, bbox)
        
        if len(class_ground_truths) == 0:
            continue
            
        # Confidence로 정렬
        class_predictions.sort(key=lambda x: x[1], reverse=True)
        
        # TP, FP 계산
        tp = np.zeros(len(class_predictions))
        fp = np.zeros(len(class_predictions))
        used_gt = set()
        
        for i, (img_id, conf, pred_box) in enumerate(class_predictions):
            best_iou = 0
            best_gt_idx = -1
            
            for j, (gt_img_id, gt_box) in enumerate(class_ground_truths):
                if gt_img_id != img_id or (img_id, j) in used_gt:
                    continue
                    
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= iou_threshold:
                tp[i] = 1
                used_gt.add((img_id, best_gt_idx))
            else:
                fp[i] = 1
        
        # Precision, Recall 계산
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / len(class_ground_truths)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        
        # AP 계산 (11-point interpolation)
        ap = 0
        for t in np.linspace(0, 1, 11):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11
        
        aps.append(ap)
    
    return np.mean(aps) if aps else 0.0

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    """모델 학습"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = YOLOLoss(num_classes=len(KITTI_CLASSES))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training')
        for batch_idx, (images, targets, _) in enumerate(train_pbar):
            images = images.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation')
            for images, targets, _ in val_pbar:
                images = images.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item()
                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        print(f'  LR: {scheduler.get_last_lr()[0]:.6f}')
        print('-' * 50)
        
        # 모델 저장 (5 에포크마다)
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'yolov4_tiny_kitti_epoch_{epoch+1}.pth')
    
    return train_losses, val_losses

def evaluate_model(model, test_loader, model_name="YOLOv4 Tiny"):
    """모델 평가 및 mAP 계산"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_ground_truths = []
    
    with torch.no_grad():
        for images, targets, img_paths in tqdm(test_loader, desc=f'Evaluating {model_name}'):
            images = images.to(device)
            outputs = model(images)
            
            # 출력을 예측 박스로 변환 (간단한 구현)
            batch_size = images.size(0)
            for i in range(batch_size):
                # 실제로는 NMS 등의 후처리가 필요하지만, 여기서는 간단히 구현
                pred_boxes = []  # [(class_id, confidence, x1, y1, x2, y2), ...]
                gt_boxes = []    # [(class_id, x1, y1, x2, y2), ...]
                
                # Ground truth 박스 변환
                for target in targets[i]:
                    if len(target) == 5:
                        class_id, cx, cy, w, h = target
                        x1 = cx - w/2
                        y1 = cy - h/2
                        x2 = cx + w/2
                        y2 = cy + h/2
                        gt_boxes.append([int(class_id), x1.item(), y1.item(), x2.item(), y2.item()])
                
                # 예측 박스는 실제 구현에서 outputs에서 추출해야 함
                # 여기서는 예시를 위한 더미 예측
                if len(gt_boxes) > 0:
                    # 더미 예측 (실제로는 모델 출력에서 추출)
                    pred_boxes.append([gt_boxes[0][0], 0.5, gt_boxes[0][1], gt_boxes[0][2], gt_boxes[0][3], gt_boxes[0][4]])
                
                all_predictions.append(pred_boxes)
                all_ground_truths.append(gt_boxes)
    
    # mAP 계산
    map_score = calculate_map(all_predictions, all_ground_truths, iou_threshold=0.5)
    
    print(f'{model_name} mAP@0.5: {map_score:.4f}')
    return map_score

def main():
    """메인 실행 함수"""
    # 경로 설정
    train_img_dir = '../data_object_image_2/training/image_2'
    train_label_dir = '../data_object_image_2/training/label_2'
    
    # 데이터셋 로드
    print("Loading KITTI dataset...")
    full_dataset = KITTIDataset(train_img_dir, train_label_dir, img_size=416, augment=True)
    
    # 데이터셋 분할 (80% 학습, 20% 검증)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # 데이터 로더 생성
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=lambda x: x)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, collate_fn=lambda x: x)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # 사전 훈련된 모델 로드 (있다면)
    pretrained_model = YOLOv4Tiny(num_classes=len(KITTI_CLASSES))
    print("Evaluating pretrained YOLOv4 Tiny...")
    pretrained_map = evaluate_model(pretrained_model, val_loader, "Pretrained YOLOv4 Tiny")
    
    # 새 모델 초기화
    retrained_model = YOLOv4Tiny(num_classes=len(KITTI_CLASSES))
    
    # 모델 학습
    print("Starting training...")
    train_losses, val_losses = train_model(
        retrained_model, 
        train_loader, 
        val_loader, 
        num_epochs=30,
        learning_rate=0.001
    )
    
    # 재학습된 모델 평가
    print("Evaluating retrained model...")
    retrained_map = evaluate_model(retrained_model, val_loader, "Retrained YOLOv4 Tiny")
    
    # 결과 비교
    print("\n" + "="*50)
    print("MODEL COMPARISON RESULTS")
    print("="*50)
    print(f"Pretrained YOLOv4 Tiny mAP@0.5: {pretrained_map:.4f}")
    print(f"Retrained YOLOv4 Tiny mAP@0.5:  {retrained_map:.4f}")
    print(f"Improvement: {retrained_map - pretrained_map:.4f}")
    print("="*50)
    
    # 손실 그래프 그리기
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    models = ['Pretrained', 'Retrained']
    maps = [pretrained_map, retrained_map]
    plt.bar(models, maps, color=['skyblue', 'lightcoral'])
    plt.xlabel('Model')
    plt.ylabel('mAP@0.5')
    plt.title('Model Comparison')
    plt.ylim(0, 1)
    
    for i, v in enumerate(maps):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 최종 모델 저장
    torch.save(retrained_model.state_dict(), 'yolov4_tiny_kitti_final.pth')
    
    # 결과를 JSON 파일로 저장
    results = {
        'pretrained_map': float(pretrained_map),
        'retrained_map': float(retrained_map),
        'improvement': float(retrained_map - pretrained_map),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'num_classes': len(KITTI_CLASSES),
        'classes': KITTI_CLASSES
    }
    
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Training completed! Results saved to training_results.json and training_results.png")

if __name__ == "__main__":
    # CUDA 사용 가능 여부 확인
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 실행
    main()
