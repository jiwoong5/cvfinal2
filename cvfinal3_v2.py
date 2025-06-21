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
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.image_dir = Path(os.path.join(base_path, image_dir))
        self.label_dir = Path(os.path.join(base_path, label_dir))
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
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 5
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Detection Head
        self.detection_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, (self.num_classes + 5) * 3, 1),  # 3 anchors per cell
        )
        
        self.img_size = 416
        self.grid_size = 13
        
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
        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5
        
    def forward(self, predictions, targets_list):
        device = predictions.device
        batch_size, num_anchors, grid_size, _, prediction_size = predictions.shape
        
        # 예측값을 평탄화
        predictions_flat = predictions.view(batch_size, -1, prediction_size)
        
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        for batch_idx in range(batch_size):
            # 각 배치에 대한 타겟
            targets = targets_list[batch_idx] if batch_idx < len(targets_list) else torch.zeros((0, 5))
            
            if len(targets) > 0:
                # 객체가 있는 경우: 좌표, 신뢰도, 클래스 손실 계산
                
                # 예측값에서 각 구성요소 추출
                pred_xy = predictions_flat[batch_idx, :, :2]      # x, y 좌표
                pred_wh = predictions_flat[batch_idx, :, 2:4]     # width, height
                pred_conf = predictions_flat[batch_idx, :, 4]     # 신뢰도
                pred_cls = predictions_flat[batch_idx, :, 5:]     # 클래스 확률
                
                # 좌표 손실 (간단한 MSE)
                coord_loss = torch.mean(pred_xy**2) + torch.mean(pred_wh**2)
                
                # 신뢰도 손실
                target_conf = torch.ones_like(pred_conf) * 0.5  # 타겟 신뢰도
                conf_loss = self.bce_loss(pred_conf, target_conf)
                
                # 클래스 손실
                target_cls = torch.zeros_like(pred_cls)
                if len(targets) > 0:
                    # 첫 번째 타겟의 클래스를 사용 (간단한 구현)
                    class_idx = int(targets[0, 0].item())
                    if class_idx < self.num_classes:
                        target_cls[:, class_idx] = 1.0
                
                cls_loss = self.bce_loss(pred_cls, target_cls)
                
                # 전체 손실
                batch_loss = (self.lambda_coord * coord_loss + 
                             conf_loss + 
                             cls_loss)
            else:
                # 객체가 없는 경우: 신뢰도만 낮추기
                pred_conf = predictions_flat[batch_idx, :, 4]
                target_conf = torch.zeros_like(pred_conf)
                batch_loss = self.lambda_noobj * self.bce_loss(pred_conf, target_conf)
            
            total_loss = total_loss + batch_loss
        
        # 배치 평균
        total_loss = total_loss / batch_size
        
        return total_loss

def non_max_suppression(predictions, conf_threshold=0.5, iou_threshold=0.4):
    """NMS(Non-Maximum Suppression) 적용"""
    filtered_predictions = []
    
    for pred_boxes in predictions:
        if len(pred_boxes) == 0:
            filtered_predictions.append([])
            continue
            
        # 신뢰도 기준 필터링
        conf_mask = [box['confidence'] >= conf_threshold for box in pred_boxes]
        filtered_boxes = [box for i, box in enumerate(pred_boxes) if conf_mask[i]]
        
        if len(filtered_boxes) == 0:
            filtered_predictions.append([])
            continue
        
        # 신뢰도로 정렬
        filtered_boxes.sort(key=lambda x: x['confidence'], reverse=True)
        
        # NMS 적용
        keep = []
        while filtered_boxes:
            current = filtered_boxes.pop(0)
            keep.append(current)
            
            # 현재 박스와 겹치는 박스들 제거
            remaining = []
            for box in filtered_boxes:
                if current['class_id'] != box['class_id']:
                    remaining.append(box)
                    continue
                    
                iou = calculate_iou(current['bbox'], box['bbox'])
                if iou <= iou_threshold:
                    remaining.append(box)
            
            filtered_boxes = remaining
        
        filtered_predictions.append(keep)
    
    return filtered_predictions

def extract_predictions_from_model_output(outputs, confidence_threshold=0.5, img_size=416):
    """모델 출력에서 예측 박스 추출 (개선된 버전)"""
    batch_size, num_anchors, grid_h, grid_w, prediction_size = outputs.shape
    
    all_predictions = []
    
    for batch_idx in range(batch_size):
        predictions = []
        
        for anchor_idx in range(num_anchors):
            for grid_y in range(grid_h):
                for grid_x in range(grid_w):
                    prediction = outputs[batch_idx, anchor_idx, grid_y, grid_x]
                    
                    # 좌표, 신뢰도, 클래스 확률 추출
                    x_center = torch.sigmoid(prediction[0])
                    y_center = torch.sigmoid(prediction[1])
                    width = prediction[2]
                    height = prediction[3]
                    confidence = torch.sigmoid(prediction[4])
                    class_probs = torch.softmax(prediction[5:], dim=0)
                    
                    # 신뢰도 필터링 (높은 임계값 사용)
                    if confidence.item() < confidence_threshold:
                        continue
                    
                    # 그리드 좌표를 이미지 좌표로 변환
                    x_center = (grid_x + x_center.item()) / grid_w
                    y_center = (grid_y + y_center.item()) / grid_h
                    width = torch.exp(width).item() / grid_w
                    height = torch.exp(height).item() / grid_h
                    
                    # 박스 좌표 계산 (정규화된 좌표)
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2
                    
                    # 이미지 크기로 스케일링
                    x1 *= img_size
                    y1 *= img_size
                    x2 *= img_size
                    y2 *= img_size
                    
                    # 클래스 예측
                    class_id = torch.argmax(class_probs).item()
                    class_confidence = class_probs[class_id].item()
                    
                    # 최종 신뢰도 = 객체 신뢰도 × 클래스 신뢰도
                    final_confidence = confidence.item() * class_confidence
                    
                    predictions.append({
                        'class_id': class_id,
                        'class_name': KITTI_CLASSES[class_id],
                        'confidence': final_confidence,
                        'bbox': [x1, y1, x2, y2]
                    })
        
        all_predictions.append(predictions)
    
    return all_predictions

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
    """mAP@0.5 계산 (class_id -1 보정 포함 + 클래스 분포 디버그 출력)"""
    aps = []
    gt_class_counter = {}
    pred_class_counter = {}

    for class_id in range(len(KITTI_CLASSES)):
        class_predictions = []
        class_ground_truths = []
        
        # 클래스별로 예측과 실제값 분리
        for img_id, (pred_boxes, gt_boxes) in enumerate(zip(predictions, ground_truths)):
            # ✅ GT 클래스 분포 수집
            for box in gt_boxes:
                if len(box) >= 5:
                    cid = int(box[0])
                    gt_class_counter[cid] = gt_class_counter.get(cid, 0) + 1
                    if cid == class_id:
                        cx, cy, w, h = box[1], box[2], box[3], box[4]
                        x1 = (cx - w/2) * 416
                        y1 = (cy - h/2) * 416
                        x2 = (cx + w/2) * 416
                        y2 = (cy + h/2) * 416
                        class_ground_truths.append((img_id, [x1, y1, x2, y2]))
            
            # ✅ 예측 클래스 분포 수집 및 보정
            for box in pred_boxes:
                corrected_cid = box['class_id']  # class_id 보정
                pred_class_counter[corrected_cid] = pred_class_counter.get(corrected_cid, 0) + 1
                if corrected_cid == class_id:
                    class_predictions.append((img_id, box['confidence'], box['bbox']))
        
        if len(class_ground_truths) == 0:
            continue
            
        class_predictions.sort(key=lambda x: x[1], reverse=True)
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
        
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recalls = tp_cumsum / len(class_ground_truths)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        
        ap = 0
        for t in np.linspace(0, 1, 11):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11
        
        aps.append(ap)
    
    # ✅ 클래스 분포 디버깅 출력
    print("\n[📊 GT 클래스 분포]:")
    for cid in sorted(gt_class_counter):
        print(f"  class {cid}: {gt_class_counter[cid]}개")

    print("\n[📊 예측 클래스 분포 (보정 x)]")
    for cid in sorted(pred_class_counter):
        print(f"  class {cid}: {pred_class_counter[cid]}개")

    return np.mean(aps) if aps else 0.0


def debug_model_weights(model, model_name):
    """모델 가중치 디버그 함수"""
    print(f"\n{'='*50}")
    print(f"DEBUG: {model_name} 모델 정보")
    print(f"{'='*50}")
    
    # 총 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"총 파라미터 수: {total_params:,}")
    print(f"학습 가능한 파라미터 수: {trainable_params:,}")
    
    # 첫 번째 레이어의 가중치 통계
    first_layer = model.backbone[0]  # 첫 번째 Conv2d 레이어
    if hasattr(first_layer, 'weight'):
        weight = first_layer.weight.data
        print(f"\n첫 번째 레이어 가중치 통계:")
        print(f"  - Shape: {weight.shape}")
        print(f"  - Mean: {weight.mean().item():.6f}")
        print(f"  - Std: {weight.std().item():.6f}")
        print(f"  - Min: {weight.min().item():.6f}")
        print(f"  - Max: {weight.max().item():.6f}")
    
    # 마지막 레이어의 가중치 통계
    last_layer = model.detection_head[-1]  # 마지막 Conv2d 레이어
    if hasattr(last_layer, 'weight'):
        weight = last_layer.weight.data
        print(f"\n마지막 레이어 가중치 통계:")
        print(f"  - Shape: {weight.shape}")
        print(f"  - Mean: {weight.mean().item():.6f}")
        print(f"  - Std: {weight.std().item():.6f}")
        print(f"  - Min: {weight.min().item():.6f}")
        print(f"  - Max: {weight.max().item():.6f}")
    
    # 가중치의 히스토그램 정보 (간단한 분포 확인)
    all_weights = []
    for param in model.parameters():
        if param.requires_grad:
            all_weights.extend(param.data.cpu().numpy().flatten())
    
    all_weights = np.array(all_weights)
    print(f"\n전체 가중치 분포:")
    print(f"  - 0에 가까운 값 (|w| < 0.001): {np.sum(np.abs(all_weights) < 0.001)} / {len(all_weights)}")
    print(f"  - 큰 값 (|w| > 1.0): {np.sum(np.abs(all_weights) > 1.0)} / {len(all_weights)}")
    
    print(f"{'='*50}\n")

def load_pretrained_model(model_path, model_name):
    """사전 훈련된 모델 로드 및 디버깅"""
    print(f"\n{'='*60}")
    print(f"Loading {model_name} from {model_path}")
    print(f"{'='*60}")
    
    # 모델 초기화
    model = YOLOv4Tiny(num_classes=len(KITTI_CLASSES))
    
    # 파일 존재 확인
    if not os.path.exists(model_path):
        print(f"ERROR: 모델 파일이 존재하지 않습니다: {model_path}")
        print("기본 초기화된 모델을 반환합니다.")
        return model
    
    try:
        # 모델 로드
        print(f"모델 파일 크기: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
        
        # CPU에서 로드 (CUDA 가용성과 관계없이)
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # checkpoint 내용 확인
        if isinstance(checkpoint, dict):
            print("Checkpoint 내용:")
            for key in checkpoint.keys():
                if isinstance(checkpoint[key], torch.Tensor):
                    print(f"  - {key}: {checkpoint[key].shape}")
                else:
                    print(f"  - {key}: {type(checkpoint[key])}")
            
            # state_dict 추출
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("state_dict에서 모델 가중치 로드")
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("model_state_dict에서 모델 가중치 로드")
            else:
                state_dict = checkpoint
                print("checkpoint를 직접 state_dict로 사용")
        else:
            state_dict = checkpoint
            print("checkpoint를 직접 state_dict로 사용")
        
        # 모델에 가중치 로드
        model.load_state_dict(state_dict, strict=False)
        print(f"✓ {model_name} 모델이 성공적으로 로드되었습니다!")
        
        # 모델 디버깅
        debug_model_weights(model, model_name)
        
    except Exception as e:
        print(f"ERROR: 모델 로드 중 오류 발생: {str(e)}")
        print("기본 초기화된 모델을 반환합니다.")
        return model
    
    return model

def evaluate_model(model, test_loader, model_name="YOLOv4 Tiny"):
    """모델 평가 및 mAP 계산 (수정된 버전)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_ground_truths = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f'Evaluating {model_name}'):
            # 배치 데이터 처리
            images = []
            targets_list = []
            img_paths = []
            
            for item in batch:
                images.append(item[0])
                targets_list.append(item[1])
                img_paths.append(item[2])
            
            # 텐서로 변환
            images = torch.stack(images).to(device)
            outputs = model(images)
            
            # 출력을 예측 박스로 변환
            predictions = extract_predictions_from_model_output(outputs, confidence_threshold=0.5)
            # NMS 적용
            predictions = non_max_suppression(predictions, conf_threshold=0.5, iou_threshold=0.4)
            
            batch_size = images.size(0)
            for i in range(batch_size):
                # 예측 박스
                pred_boxes = predictions[i] if i < len(predictions) else []
                
                # Ground truth 박스
                targets = targets_list[i]
                gt_boxes = []
                for target in targets:
                    if len(target) == 5:
                        gt_boxes.append(target.tolist())  # [class_id, cx, cy, w, h]
                
                all_predictions.append(pred_boxes)
                all_ground_truths.append(gt_boxes)
    
    # mAP 계산
    map_score = calculate_map(all_predictions, all_ground_truths, iou_threshold=0.5)
    
    print(f'{model_name} mAP@0.5: {map_score:.4f}')
    
    # 상세 통계 출력
    total_predictions = sum(len(preds) for preds in all_predictions)
    total_ground_truths = sum(len(gts) for gts in all_ground_truths)
    
    print(f'{model_name} 상세 통계:')
    print(f'  총 예측 수: {total_predictions}')
    print(f'  총 실제 객체 수: {total_ground_truths}')
    print(f'  평균 예측/이미지: {total_predictions/len(all_predictions):.2f}')
    print(f'  평균 실제 객체/이미지: {total_ground_truths/len(all_ground_truths):.2f}')
    
    return map_score

def collate_fn(batch):
    """커스텀 collate 함수"""
    return batch

def get_num_classes_from_state_dict(state_dict):
    # 예시: 클래스 수 추론 (YOLOv4-tiny의 출력 채널 = num_anchors × (5 + num_classes))
    for k, v in state_dict.items():
        if "head" in k and "weight" in k:
            out_channels = v.size(0)
            num_anchors = 3  # 보통 YOLO 한 scale에 3 anchor
            num_classes = out_channels // num_anchors - 5
            return num_classes
    return 3  # fallback

#모델 로드 함수
def load_model(weight_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    state_dict = torch.load(weight_path, map_location=device)
    num_classes = get_num_classes_from_state_dict(state_dict)
    print(f"Auto-detected class count: {num_classes}")
    model = YOLOv4Tiny(num_classes=num_classes)
    model.load_state_dict(state_dict)  # strict=True 기본값
    model.to(device)
    model.eval()
    return model

def main():
    """메인 실행 함수"""
    # 경로 설정
    train_img_dir = '../data_object_image_2/training/image_2'
    train_label_dir = '../training/label_2'
    
    # 데이터셋 로드
    print("Loading KITTI dataset...")
    full_dataset = KITTIDataset(train_img_dir, train_label_dir, img_size=416, augment=True)
    
    # 데이터셋 분할 (80% 학습, 20% 검증)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # 데이터 로더 생성 (num_workers=0으로 설정하여 multiprocessing 문제 해결)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # 사전 훈련된 모델 로드 (있다면)
    pretrained_model = YOLOv4Tiny(num_classes=len(KITTI_CLASSES))
    print("Evaluating pretrained original YOLOv4 Tiny...")
    pretrained_map = evaluate_model(pretrained_model, val_loader, "Pretrained YOLOv4 Tiny")

    # Biased 모델
    biased_model = load_pretrained_model('./models/biased_model.pth', 'Biased YOLOv4 Tiny')
    print("Evaluating pretrained by biased_model YOLOv4 Tiny...")
    pretrained_biased_map = evaluate_model(biased_model, val_loader, "Pretrained YOLOv4 Tiny by biased_model YOLOv4 Tiny")

    # Balanced 모델
    balanced_model = load_pretrained_model('./models/balanced_model.pth', 'Balanced YOLOv4 Tiny')
    print("Evaluating pretrained by balanced_model YOLOv4 Tiny...")
    pretrained_balanced_map = evaluate_model(balanced_model, val_loader, "Pretrained YOLOv4 Tiny by balanced_model YOLOv4 Tiny")

    '''
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
    '''

    # 결과 비교
    print("\n" + "="*50)
    print("MODEL COMPARISON RESULTS")
    print("="*50)
    print(f"Pretrained YOLOv4 Tiny mAP@0.5: {pretrained_map:.4f}")
    print(f"Retrained by biased train set YOLOv4 Tiny mAP@0.5:  {pretrained_biased_map:.4f}")
    print(f"Retrained by balanced train set YOLOv4 Tiny mAP@0.5:  {pretrained_balanced_map:.4f}")
    print(f"Improvement pretrained_biased_map - pretrained_map: {pretrained_biased_map - pretrained_map:.4f}")
    print(f"Improvement pretrained_balanced_map - pretrained_map: {pretrained_balanced_map - pretrained_map:.4f}")
    print("="*50)
    '''
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
    '''

    # 모델 성능 비교 바 차트
    plt.figure(figsize=(6, 5))
    models = ['Pretrained', 'Biased', 'Balanced']
    maps = [pretrained_map, pretrained_biased_map, pretrained_balanced_map]
    colors = ['skyblue', 'salmon', 'lightgreen']

    plt.bar(models, maps, color=colors)
    plt.ylim(0, 1)
    plt.title('YOLOv4 Tiny Model Comparison (mAP@0.5)')
    plt.ylabel('mAP@0.5')

    for i, v in enumerate(maps):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 최종 모델 저장
    #torch.save(retrained_model.state_dict(), 'yolov4_tiny_kitti_final.pth')
    
    # 결과를 JSON 파일로 저장
    results = {
        'pretrained_map': float(pretrained_map),
        'biased_map': float(pretrained_biased_map),
        'balanced_map': float(pretrained_balanced_map),
        'improvement_biased': float(pretrained_biased_map - pretrained_map),
        'improvement_balanced': float(pretrained_balanced_map - pretrained_map),
        'num_classes': len(KITTI_CLASSES),
        'classes': KITTI_CLASSES
    }

    with open('model_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Evaluation completed! Results saved to model_comparison_results.json and model_comparison_results.png")

if __name__ == "__main__":
    main()