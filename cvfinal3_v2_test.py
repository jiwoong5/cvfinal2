import glob
import random
import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.models as models

# -------------------------------
# 1. KITTI -> COCO 클래스 매핑
# -------------------------------
KITTI_TO_COCO = {
    'Car': 'car',
    'Van': 'car',
    'Truck': 'truck',
    'Pedestrian': 'person',
    'Person_sitting': 'person',
    'Cyclist': 'bicycle',
    'Tram': 'train',
}

KITTI_CLASSES = [
    'Car', 'Van', 'Truck', 'Pedestrian',
    'Person_sitting', 'Cyclist', 'Tram', 'Misc'
]

# COCO 클래스 ID 매핑 (PyTorch 모델용)
COCO_CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))
    
# -------------------------------
# 2. PyTorch 모델 정의
# -------------------------------
class SimpleDetector(nn.Module):
    """간단한 객체 검출 모델 (예시)"""
    def __init__(self, num_classes=80):
        super(SimpleDetector, self).__init__()
        # ResNet 백본 사용 - 경고 메시지 해결
        self.backbone = models.resnet50(weights=None)  # pretrained=False 대신 weights=None 사용
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 512)
        
        # 검출 헤드
        self.classifier = nn.Linear(512, num_classes)
        self.bbox_regressor = nn.Linear(512, 4)  # x1,y1,x2,y2
        self.confidence_head = nn.Linear(512, 1)
        
    def forward(self, x):
        features = self.backbone(x)
        
        class_scores = self.classifier(features)
        bbox_pred = self.bbox_regressor(features)
        confidence = torch.sigmoid(self.confidence_head(features))
        
        return {
            'class_scores': class_scores,
            'bbox_pred': bbox_pred,
            'confidence': confidence
        }

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

# -------------------------------
# 3. PyTorch 모델 추론 함수
# -------------------------------
def pytorch_inference(model, image, device, conf_thresh=0.3):
    """PyTorch YOLO 모델로 추론 - KITTI 전용"""
    model.eval()
    
    # 이미지 전처리
    transform = transforms.Compose([
        transforms.Resize((416, 416)),  # YOLO 입력 크기
        transforms.ToTensor(),
    ])
    
    # OpenCV BGR -> RGB 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # 원본 이미지 크기
    orig_h, orig_w = image.shape[:2]
    
    input_tensor = transform(pil_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)  # out1, out2, out3
        
        # YOLO 출력 처리
        all_boxes = []
        all_scores = []
        all_class_ids = []
        
        # 각 스케일별로 처리
        for scale_idx, output in enumerate(outputs):
            batch_size, channels, grid_h, grid_w = output.shape
            anchors_per_scale = 3
            num_classes = len(KITTI_CLASSES)
            
            # 출력을 (batch, anchors, grid_h, grid_w, 5+classes)로 reshape
            output = output.view(batch_size, anchors_per_scale, num_classes + 5, grid_h, grid_w)
            output = output.permute(0, 1, 3, 4, 2).contiguous()  # (batch, anchors, grid_h, grid_w, 5+classes)
            
            # 시그모이드 적용
            xy = torch.sigmoid(output[..., 0:2])  # Center x, y offset
            wh = output[..., 2:4]  # Width, height
            conf = torch.sigmoid(output[..., 4:5])  # Objectness
            class_pred = torch.sigmoid(output[..., 5:])  # Class probabilities
            
            # 그리드 생성
            grid_x = torch.arange(grid_w, device=device, dtype=torch.float32).repeat(grid_h, 1).view(1, 1, grid_h, grid_w)
            grid_y = torch.arange(grid_h, device=device, dtype=torch.float32).repeat(grid_w, 1).t().view(1, 1, grid_h, grid_w)
            
            # 절대 좌표 계산
            stride = 416 // grid_h  # 입력 크기에서 현재 그리드 크기로 나눈 stride
            pred_boxes = torch.zeros_like(output[..., :4])
            pred_boxes[..., 0] = (xy[..., 0] + grid_x) * stride  # center x
            pred_boxes[..., 1] = (xy[..., 1] + grid_y) * stride  # center y  
            pred_boxes[..., 2] = torch.exp(wh[..., 0]) * stride  # width
            pred_boxes[..., 3] = torch.exp(wh[..., 1]) * stride  # height
            
            # 신뢰도 필터링
            conf_mask = (conf.squeeze(-1) > conf_thresh)  # (batch, anchors, grid_h, grid_w)
            
            for b in range(batch_size):  # 배치별 처리
                for a in range(anchors_per_scale):  # 앵커별 처리
                    mask = conf_mask[b, a]  # (grid_h, grid_w)
                    if mask.sum() == 0:
                        continue
                        
                    # 마스크된 예측 추출
                    valid_boxes = pred_boxes[b, a][mask]  # (N, 4)
                    valid_conf = conf[b, a][mask]  # (N, 1)
                    valid_class_pred = class_pred[b, a][mask]  # (N, num_classes)
                    
                    # Center, width, height -> x1, y1, x2, y2 변환
                    x1 = valid_boxes[:, 0] - valid_boxes[:, 2] / 2
                    y1 = valid_boxes[:, 1] - valid_boxes[:, 3] / 2
                    x2 = valid_boxes[:, 0] + valid_boxes[:, 2] / 2
                    y2 = valid_boxes[:, 1] + valid_boxes[:, 3] / 2
                    
                    # 원본 이미지 크기로 스케일링
                    x1 = x1 * orig_w / 416
                    y1 = y1 * orig_h / 416  
                    x2 = x2 * orig_w / 416
                    y2 = y2 * orig_h / 416
                    
                    # 박스 좌표 클리핑
                    x1 = torch.clamp(x1, 0, orig_w)
                    y1 = torch.clamp(y1, 0, orig_h)
                    x2 = torch.clamp(x2, 0, orig_w)
                    y2 = torch.clamp(y2, 0, orig_h)
                    
                    boxes = torch.stack([x1, y1, x2, y2], dim=1)
                    
                    # 클래스 점수와 ID
                    class_scores, class_ids = torch.max(valid_class_pred, dim=1)
                    final_scores = valid_conf.squeeze() * class_scores
                    
                    if len(boxes) > 0:
                        all_boxes.append(boxes.cpu().numpy())
                        all_scores.append(final_scores.cpu().numpy())
                        all_class_ids.append(class_ids.cpu().numpy())
        
        # 모든 스케일 결과 합치기
        if len(all_boxes) > 0:
            final_boxes = np.concatenate(all_boxes, axis=0)
            final_scores = np.concatenate(all_scores, axis=0)
            final_class_ids = np.concatenate(all_class_ids, axis=0)
            
            return final_boxes, final_scores, final_class_ids
        else:
            return np.array([]), np.array([]), np.array([])

# -------------------------------
# 4. IoU 함수
# -------------------------------
def iou(boxA, boxB):
    # box = [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    denom = boxAArea + boxBArea - interArea
    if denom == 0:
        return 0
    return interArea / denom

# -------------------------------
# 5. AP 계산 함수 (11-point interpolation)
# -------------------------------
def voc_ap(recalls, precisions):
    # 11 point interpolation
    ap = 0.0
    for t in np.arange(0., 1.1, 0.1):
        p = 0
        for i in range(len(recalls)):
            if recalls[i] >= t:
                p = max(p, precisions[i])
        ap += p / 11.0
    return ap

# -------------------------------
# 6. GT 박스 로드 함수
# -------------------------------
def load_gt_boxes(label_path, class_names):
    boxes = []
    if not os.path.exists(label_path):
        print(f"Warning: Label file not found: {label_path}")
        return boxes
    
    try:
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 8:
                    continue
                    
                kitti_class = parts[0]
                # KITTI 클래스 직접 사용 (매핑 없이)
                if kitti_class not in KITTI_CLASSES:
                    continue
                    
                # KITTI 라벨 포맷: class truncated occluded alpha x1 y1 x2 y2 ...
                x1 = int(float(parts[4]))
                y1 = int(float(parts[5]))
                x2 = int(float(parts[6]))
                y2 = int(float(parts[7]))
                boxes.append(([x1, y1, x2, y2], kitti_class))
    except Exception as e:
        print(f"Error reading label file {label_path}: {e}")
        
    return boxes

# -------------------------------
# 7. 이미지별 매칭 요약 함수
# -------------------------------
def print_image_matching_summary(image_id, predictions, gt_boxes, model_name, iou_thresh=0.5):
    """이미지별 예측-GT 매칭 결과 요약"""
    
    print(f"\n📋 Matching Summary for {image_id} ({model_name}):")
    print("-" * 60)
    
    if not predictions and not gt_boxes:
        print("No predictions and no GT objects")
        return
    
    # 클래스별 통계
    pred_by_class = {}
    gt_by_class = {}
    
    # 예측 객체 분류
    for pred in predictions:
        cls = pred['class']
        if cls not in pred_by_class:
            pred_by_class[cls] = []
        pred_by_class[cls].append(pred)
    
    # GT 객체 분류  
    for gt in gt_boxes:
        cls = gt[1]  # (box, class_name)
        if cls not in gt_by_class:
            gt_by_class[cls] = []
        gt_by_class[cls].append(gt)
    
    # 전체 클래스
    all_classes = set(list(pred_by_class.keys()) + list(gt_by_class.keys()))
    
    for cls in sorted(all_classes):
        preds = pred_by_class.get(cls, [])
        gts = gt_by_class.get(cls, [])
        
        print(f"\n🏷️  Class: {cls}")
        print(f"   Predictions: {len(preds)}, GT: {len(gts)}")
        
        if preds and gts:
            # IoU 매칭 수행
            matched_pairs = []
            for i, pred in enumerate(preds):
                best_iou = 0
                best_gt_idx = -1
                
                for j, (gt_box, gt_cls) in enumerate(gts):
                    iou_score = iou(pred['box'], gt_box) 
                    if iou_score > best_iou:
                        best_iou = iou_score
                        best_gt_idx = j
                
                if best_iou >= iou_thresh:
                    matched_pairs.append((i, best_gt_idx, best_iou))
                    print(f"   ✅ PRED-{i+1} ↔ GT-{best_gt_idx+1}: IoU = {best_iou:.4f}")
                else:
                    print(f"   ❌ PRED-{i+1}: No match (best IoU = {best_iou:.4f})")
            
            # 매칭되지 않은 GT
            matched_gt_indices = {pair[1] for pair in matched_pairs}
            for j in range(len(gts)):
                if j not in matched_gt_indices:
                    print(f"   🔍 GT-{j+1}: Missed (False Negative)")
        
        elif preds and not gts:
            print(f"   ❌ All {len(preds)} predictions are False Positives")
        elif not preds and gts:
            print(f"   🔍 All {len(gts)} GT objects are False Negatives")

# -------------------------------
# 8. mAP 계산 함수
# -------------------------------
def compute_map(all_detections, all_annotations, model_name, iou_thresh=0.5):
    """
    all_detections: dict[class] = list of (image_id, confidence, box)
    all_annotations: dict[class] = dict[image_id] = list of (box, class_name)
    """
    APs = []
    
    # 검출된 클래스만 처리
    valid_classes = [cls for cls in all_detections.keys() 
                    if len(all_detections[cls]) > 0 or len(all_annotations[cls]) > 0]
    
    print(f"\n📊 Computing mAP for {model_name}...")
    
    for cls in valid_classes:
        detections = sorted(all_detections[cls], key=lambda x: x[1], reverse=True)  # confidence 내림차순
        gt_per_image = all_annotations[cls]  # dict[img_id] = list of (box, class_name)

        # GT가 없으면 스킵
        total_gt = sum([len(gt_per_image[img_id]) for img_id in gt_per_image])
        if total_gt == 0:
            print(f"Class '{cls}': No GT boxes found, skipping...")
            continue

        TP = np.zeros(len(detections))
        FP = np.zeros(len(detections))

        # GT 박스 매칭 기록
        matched = {img_id: np.zeros(len(gt_per_image[img_id])) for img_id in gt_per_image}

        for i, (img_id, conf, det_box) in enumerate(detections):
            gt_items = gt_per_image.get(img_id, [])
            
            if len(gt_items) == 0:
                FP[i] = 1
                continue
                
            gt_boxes = [gt[0] for gt in gt_items]
            ious = np.array([iou(det_box, gt_box) for gt_box in gt_boxes])
            
            max_iou_idx = np.argmax(ious)
            max_iou = ious[max_iou_idx]

            if max_iou >= iou_thresh:
                if matched[img_id][max_iou_idx] == 0:
                    TP[i] = 1
                    matched[img_id][max_iou_idx] = 1
                else:
                    FP[i] = 1
            else:
                FP[i] = 1

        cum_TP = np.cumsum(TP)
        cum_FP = np.cumsum(FP)
        recalls = cum_TP / (total_gt + 1e-6)
        precisions = cum_TP / (cum_TP + cum_FP + 1e-6)

        ap = voc_ap(recalls, precisions)
        APs.append(ap)

        print(f"  Class '{cls}': Total GT: {total_gt}, Total Det: {len(detections)}, AP = {ap:.4f}")

    mAP = np.mean(APs) if len(APs) > 0 else 0
    print(f"\n📈 {model_name} mAP @ IoU {iou_thresh}: {mAP:.4f}")
    return mAP

# -------------------------------
# 9. 파일 확인 함수
# -------------------------------
def check_files():
    """필요한 파일들이 존재하는지 확인"""
    yolo_files = {
        'cfg': 'yolov4-tiny.cfg',
        'weights': 'yolov4-tiny.weights', 
        'names': 'coco.names'
    }
    
    pytorch_files = {
        'biased': './models/biased_model.pth',
        'balanced': './models/balanced_model.pth'
    }
    
    # YOLO 파일 확인
    missing_yolo = []
    for file_type, filename in yolo_files.items():
        if not os.path.exists(filename):
            missing_yolo.append(filename)
    
    # PyTorch 모델 파일 확인
    missing_pytorch = []
    for model_type, filepath in pytorch_files.items():
        if not os.path.exists(filepath):
            missing_pytorch.append(filepath)
    
    if missing_yolo:
        print("⚠️ Missing YOLO files (YOLO evaluation will be skipped):")
        for file in missing_yolo:
            print(f"  - {file}")
    
    if missing_pytorch:
        print("⚠️ Missing PyTorch model files (PyTorch evaluation will be skipped):")
        for file in missing_pytorch:
            print(f"  - {file}")
    
    return len(missing_yolo) == 0, len(missing_pytorch) == 0

def load_pytorch_model_flexibly(model_path, device):
    """유연한 PyTorch 모델 로딩 - 여러 모델 구조를 시도"""
    
    # 먼저 저장된 모델의 키들을 확인
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 모델의 키들을 확인하여 구조 파악
        keys = list(state_dict.keys())
        print(f"📋 Model keys preview: {keys[:5]}...")
        
        # 클래스 수 추정
        num_classes = estimate_num_classes(state_dict)
        print(f"🔍 Estimated num_classes: {num_classes}")
        
    except Exception as e:
        print(f"❌ Error reading model file: {e}")
        return None
    
    # 다양한 모델 구조로 시도
    model_classes = [
        ('YOLOv4Tiny', lambda: YOLOv4Tiny(num_classes=num_classes)),
        ('SimpleDetector', lambda: SimpleDetector(num_classes=num_classes)),
        ('YOLOv4Tiny_COCO', lambda: YOLOv4Tiny(num_classes=80)),  # COCO 클래스 수
        ('SimpleDetector_COCO', lambda: SimpleDetector(num_classes=80)),
        ('YOLOv4Tiny_KITTI', lambda: YOLOv4Tiny(num_classes=len(KITTI_CLASSES))),
        ('SimpleDetector_KITTI', lambda: SimpleDetector(num_classes=len(KITTI_CLASSES))),
    ]
    
    for model_name, model_factory in model_classes:
        try:
            print(f"🔄 Trying to load as {model_name}...")
            model = model_factory()
            model.load_state_dict(state_dict, strict=False)  # strict=False로 부분 로딩 허용
            model.to(device)
            model.eval()
            print(f"✅ Successfully loaded as {model_name}")
            return model, num_classes
        except Exception as e:
            print(f"❌ Failed to load as {model_name}: {str(e)[:100]}...")
            continue
    
    # 모든 시도가 실패한 경우, 커스텀 모델 생성 시도
    try:
        print("🔧 Attempting to create custom model...")
        model = create_custom_model_from_state_dict(state_dict, device)
        if model is not None:
            print("✅ Successfully created custom model")
            return model, num_classes
    except Exception as e:
        print(f"❌ Custom model creation failed: {e}")
    
    return None, 0


def estimate_num_classes(state_dict):
    """state_dict에서 클래스 수 추정"""
    # 일반적인 분류 레이어 이름들
    classifier_keys = ['classifier.weight', 'fc.weight', 'head.weight', 'output.weight']
    
    for key in classifier_keys:
        if key in state_dict:
            return state_dict[key].shape[0]
    
    # YOLO 스타일 출력 레이어 찾기
    for key in state_dict.keys():
        if 'conv' in key.lower() and 'weight' in key and len(state_dict[key].shape) == 4:
            # 마지막 conv 레이어일 가능성이 높음
            out_channels = state_dict[key].shape[0]
            # YOLO 형식: (classes + 5) * anchors
            if out_channels % 5 == 0:
                return out_channels // 5 - 1
            elif out_channels % 6 == 0:
                return out_channels // 6 - 1
    
    # 기본값 반환
    return len(KITTI_CLASSES)


def create_custom_model_from_state_dict(state_dict, device):
    """state_dict를 기반으로 커스텀 모델 생성"""
    
    class FlexibleModel(nn.Module):
        def __init__(self, state_dict):
            super().__init__()
            self.layers = nn.ModuleDict()
            
            # state_dict에서 레이어 구조 추론
            for key, param in state_dict.items():
                if 'weight' in key:
                    layer_name = key.replace('.weight', '')
                    if len(param.shape) == 4:  # Conv2d
                        in_channels, out_channels = param.shape[1], param.shape[0]
                        kernel_size = param.shape[2]
                        self.layers[layer_name] = nn.Conv2d(in_channels, out_channels, kernel_size)
                    elif len(param.shape) == 2:  # Linear
                        in_features, out_features = param.shape[1], param.shape[0]
                        self.layers[layer_name] = nn.Linear(in_features, out_features)
        
        def forward(self, x):
            # 간단한 순방향 패스
            for name, layer in self.layers.items():
                if isinstance(layer, nn.Conv2d):
                    x = F.relu(layer(x))
                elif isinstance(layer, nn.Linear):
                    x = layer(x.view(x.size(0), -1))
            return x
    
    try:
        model = FlexibleModel(state_dict)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        return model
    except:
        return None


def pytorch_inference_flexible(model, image, device, conf_thresh=0.3, num_classes=None):
    """유연한 PyTorch 추론 함수"""
    try:
        # 이미지 전처리
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (416, 416))
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        # 모델 추론
        with torch.no_grad():
            outputs = model(image_tensor)
        
        # 출력 형태에 따라 다르게 처리
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # 출력 차원 확인
        if outputs.dim() == 4:  # (batch, channels, height, width)
            outputs = outputs.squeeze(0)  # batch 차원 제거
        
        # 결과 파싱
        boxes, confidences, class_ids = parse_model_output(outputs, image.shape, conf_thresh, num_classes)
        
        return boxes, confidences, class_ids
        
    except Exception as e:
        print(f"Inference error: {e}")
        return [], [], []


def parse_model_output(outputs, original_shape, conf_thresh, num_classes):
    """모델 출력을 파싱하여 박스, 신뢰도, 클래스 ID 추출"""
    boxes, confidences, class_ids = [], [], []
    
    try:
        # 출력 형태에 따라 다르게 처리
        if outputs.dim() == 3:  # (channels, height, width)
            # YOLO 스타일 출력 처리
            for i in range(outputs.shape[1]):
                for j in range(outputs.shape[2]):
                    # 각 그리드 셀에서 예측값 추출
                    predictions = outputs[:, i, j]
                    
                    # 신뢰도 확인
                    if len(predictions) >= 5:
                        confidence = predictions[4].item()
                        if confidence > conf_thresh:
                            # 박스 좌표 계산
                            x_center = predictions[0].item() * original_shape[1]
                            y_center = predictions[1].item() * original_shape[0]
                            width = predictions[2].item() * original_shape[1]
                            height = predictions[3].item() * original_shape[0]
                            
                            x1 = int(x_center - width / 2)
                            y1 = int(y_center - height / 2)
                            x2 = int(x_center + width / 2)
                            y2 = int(y_center + height / 2)
                            
                            # 클래스 예측
                            if len(predictions) > 5:
                                class_scores = predictions[5:]
                                class_id = torch.argmax(class_scores).item()
                                
                                if num_classes is None or class_id < num_classes:
                                    boxes.append([x1, y1, x2, y2])
                                    confidences.append(confidence)
                                    class_ids.append(class_id)
        
        elif outputs.dim() == 2:  # (num_detections, prediction_values)
            # 직접적인 감지 결과 형태
            for detection in outputs:
                if len(detection) >= 6:
                    confidence = detection[4].item()
                    if confidence > conf_thresh:
                        x1, y1, x2, y2 = detection[:4].int().tolist()
                        class_id = detection[5].int().item()
                        
                        if num_classes is None or class_id < num_classes:
                            boxes.append([x1, y1, x2, y2])
                            confidences.append(confidence)
                            class_ids.append(class_id)
        
        # NMS 적용
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            confidences = torch.tensor(confidences, dtype=torch.float32)
            class_ids = torch.tensor(class_ids, dtype=torch.int64)
            
            # torchvision NMS 사용
            try:
                from torchvision.ops import nms
                keep = nms(boxes, confidences, 0.4)
                boxes = boxes[keep]
                confidences = confidences[keep]
                class_ids = class_ids[keep]
            except ImportError:
                # torchvision이 없는 경우 간단한 필터링
                pass
    
    except Exception as e:
        print(f"Output parsing error: {e}")
    
    return boxes, confidences, class_ids

# -------------------------------
# 10. 메인 함수
# -------------------------------
def main():
    print("🚀 Starting Multi-Model Object Detection Evaluation...")
    
    # 파일 존재 확인
    yolo_available, pytorch_available = check_files()
    
    if not yolo_available and not pytorch_available:
        print("❌ No models available for evaluation!")
        return
    
    # GPU 사용 가능 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # 이미지 경로 설정
    possible_image_paths = [
        '../data_object_image_2/training/image_2/*.png',
        './data_object_image_2/training/image_2/*.png',
        './images/*.png',
        './image_2/*.png',
        '*.png'
    ]
    
    image_paths = []
    for path_pattern in possible_image_paths:
        image_paths = glob.glob(path_pattern)
        if image_paths:
            print(f"✅ Found {len(image_paths)} images in {path_pattern}")
            break
    
    if not image_paths:
        print("❌ No images found! Please check your image directory path.")
        return
    
    # 샘플링
    num_samples = min(10, len(image_paths))
    sampled_paths = random.sample(image_paths, num_samples)
    print(f"📸 Processing {num_samples} sample images...")

    # 모델들 및 결과 저장
    models_results = {}
    
    # ========================================
    # YOLO 모델 평가
    # ========================================
    if yolo_available:
        print(f"\n{'='*60}")
        print("🎯 Evaluating YOLO Model")
        print(f"{'='*60}")
        
        # COCO 클래스 이름 로드
        with open('coco.names', 'r') as f:
            classes = f.read().strip().split('\n')
        
        # YOLO 모델 로딩
        net = cv2.dnn.readNetFromDarknet('yolov4-tiny.cfg', 'yolov4-tiny.weights')
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
        
        # YOLO 결과 저장
        yolo_detections = {cls: [] for cls in classes}
        yolo_annotations = {cls: {} for cls in classes}
        
        # 이미지 처리
        for idx, path in enumerate(sampled_paths):
            print(f"\n[YOLO] Processing image {idx+1}/{len(sampled_paths)}: {os.path.basename(path)}")
            
            image = cv2.imread(path)
            if image is None:
                continue
                
            height, width = image.shape[:2]
            
            # YOLO 추론
            blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            outputs = net.forward(output_layers)
            
            boxes, confidences, class_ids = [], [], []
            
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = int(np.argmax(scores))
                    confidence = scores[class_id]
                    
                    if confidence > 0.3:
                        center_x, center_y, w, h = detection[0:4] * np.array([width, height, width, height])
                        x1 = int(center_x - w / 2)
                        y1 = int(center_y - h / 2)
                        x2 = int(center_x + w / 2)
                        y2 = int(center_y + h / 2)
                        
                        boxes.append([x1, y1, x2, y2])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # NMS 적용
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
            if isinstance(indices, tuple) or len(indices) == 0:
                indices = []
            else:
                indices = indices.flatten()
            
            # GT 로드
            image_id = os.path.splitext(os.path.basename(path))[0]
            possible_label_paths = [
                f'../training/label_2/{image_id}.txt',
                f'./training/label_2/{image_id}.txt',
                f'./labels/{image_id}.txt',
                f'./label_2/{image_id}.txt',
                f'./{image_id}.txt'
            ]
            
            label_path = None
            for lp in possible_label_paths:
                if os.path.exists(lp):
                    label_path = lp
                    break
            
            gt_boxes = load_gt_boxes(label_path, classes) if label_path else []
            
            # GT 데이터 저장
            for box, class_name in gt_boxes:
                if image_id not in yolo_annotations[class_name]:
                    yolo_annotations[class_name][image_id] = []
                yolo_annotations[class_name][image_id].append((box, class_name))
            
            # 예측 데이터 저장
            for i in indices:
                cls_name = classes[class_ids[i]]
                yolo_detections[cls_name].append((image_id, confidences[i], boxes[i]))
            
            # 상세 출력
            print_detailed_results(image_id, indices, boxes, confidences, class_ids, classes, gt_boxes, "YOLO")
        
        # YOLO mAP 계산
        yolo_map = compute_map(yolo_detections, yolo_annotations, "YOLO", iou_thresh=0.5)
        models_results['YOLO'] = yolo_map
    
    # ========================================
    # PyTorch 모델들 평가 (유연한 로딩 사용)
    # ========================================
    if pytorch_available:
        pytorch_results = evaluate_pytorch_models_flexible(sampled_paths, device)
        models_results.update(pytorch_results)
    
    # ========================================
    # 최종 결과 비교
    # ========================================
    print(f"\n{'='*80}")
    print("📊 Final Results Comparison")
    print(f"{'='*80}")
    
    if models_results:
        # 결과 정렬 (mAP 기준)
        sorted_results = sorted(models_results.items(), key=lambda x: x[1], reverse=True)
        
        print("\n🏆 Model Performance Ranking:")
        for rank, (model_name, map_score) in enumerate(sorted_results, 1):
            print(f"{rank}. {model_name}: {map_score:.4f} mAP")
        
        # 성능 차이 분석
        if len(sorted_results) >= 2:
            best_model, best_score = sorted_results[0]
            worst_model, worst_score = sorted_results[-1]
            performance_gap = best_score - worst_score
            
            print(f"\n📈 Performance Analysis:")
            print(f"   Best: {best_model} ({best_score:.4f})")
            print(f"   Worst: {worst_model} ({worst_score:.4f})")
            print(f"   Gap: {performance_gap:.4f} ({performance_gap/worst_score*100:.1f}% improvement)")
    else:
        print("❌ No evaluation results available!")
    
    print(f"\n{'='*80}")
    print("✅ Evaluation Complete!")
    print(f"{'='*80}")

def print_detailed_results(image_id, indices, boxes, confidences, class_ids, classes, gt_boxes, model_name):
    """YOLO 모델의 상세 결과 출력"""
    
    print(f"\n📸 Image: {image_id} ({model_name})")
    print("-" * 50)
    
    # 예측 결과 출력
    if len(indices) > 0:
        print(f"🎯 Predictions ({len(indices)}):")
        for i in indices:
            cls_name = classes[class_ids[i]]
            conf = confidences[i]
            box = boxes[i]
            print(f"  - {cls_name}: {conf:.3f} | Box: [{box[0]}, {box[1]}, {box[2]}, {box[3]}]")
    else:
        print("🎯 Predictions: None")
    
    # GT 출력
    if gt_boxes:
        print(f"🏷️  Ground Truth ({len(gt_boxes)}):")
        for i, (box, cls_name) in enumerate(gt_boxes):
            print(f"  - {cls_name}: Box: [{box[0]}, {box[1]}, {box[2]}, {box[3]}]")
    else:
        print("🏷️  Ground Truth: None")
    
    # 매칭 결과 출력
    if len(indices) > 0 and gt_boxes:
        predictions = []
        for i in indices:
            predictions.append({
                'class': classes[class_ids[i]],
                'confidence': confidences[i],
                'box': boxes[i]
            })
        
        print_image_matching_summary(image_id, predictions, gt_boxes, model_name)

def print_detailed_pytorch_results(image_id, boxes, confidences, class_ids, class_names, gt_boxes, model_name):
    """PyTorch 모델의 상세 결과 출력"""
    
    print(f"\n📸 Image: {image_id} ({model_name})")
    print("-" * 50)
    
    # 예측 결과 출력
    if len(boxes) > 0:
        print(f"🎯 Predictions ({len(boxes)}):")
        for i in range(len(boxes)):
            cls_name = class_names[class_ids[i]]
            conf = confidences[i]
            box = boxes[i].astype(int)
            print(f"  - {cls_name}: {conf:.3f} | Box: [{box[0]}, {box[1]}, {box[2]}, {box[3]}]")
    else:
        print("🎯 Predictions: None")
    
    # GT 출력
    if gt_boxes:
        print(f"🏷️  Ground Truth ({len(gt_boxes)}):")
        for i, (box, cls_name) in enumerate(gt_boxes):
            print(f"  - {cls_name}: Box: [{box[0]}, {box[1]}, {box[2]}, {box[3]}]")
    else:
        print("🏷️  Ground Truth: None")
    
    # 매칭 결과 출력
    if len(boxes) > 0 and gt_boxes:
        predictions = []
        for i in range(len(boxes)):
            predictions.append({
                'class': class_names[class_ids[i]],
                'confidence': confidences[i],
                'box': boxes[i].astype(int)
            })
        
        print_image_matching_summary(image_id, predictions, gt_boxes, model_name)

def load_pytorch_model(model_path, device):
    """PyTorch 모델 로딩"""
    try:
        # KITTI 클래스 수로 모델 생성
        model = YOLOv4Tiny(num_classes=len(KITTI_CLASSES))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        print(f"✅ Loaded model as YOLOv4Tiny with {len(KITTI_CLASSES)} KITTI classes")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None

def evaluate_pytorch_models(sampled_paths, device):
    """PyTorch 모델들 평가 (유연한 로딩 사용)"""
    pytorch_models = {
        'Biased Model': './models/biased_model.pth',
        'Balanced Model': './models/balanced_model.pth'
    }
    
    results = {}
    
    for model_name, model_path in pytorch_models.items():
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            continue
            
        print(f"\n{'='*60}")
        print(f"🧠 Evaluating {model_name}")
        print(f"{'='*60}")
        
        # 유연한 모델 로드
        model_result = load_pytorch_model_flexibly(model_path, device)
        if model_result is None:
            print(f"❌ Failed to load {model_name}")
            continue
        
        model, num_classes = model_result
        
        # 결과 저장 - KITTI 클래스 사용
        pytorch_detections = {cls: [] for cls in KITTI_CLASSES}
        pytorch_annotations = {cls: {} for cls in KITTI_CLASSES}
        
        # 이미지 처리
        for idx, path in enumerate(sampled_paths):
            print(f"\n[{model_name}] Processing image {idx+1}/{len(sampled_paths)}: {os.path.basename(path)}")
            
            image = cv2.imread(path)
            if image is None:
                continue
            
            # 유연한 PyTorch 추론
            try:
                boxes, confidences, class_ids = pytorch_inference_flexible(model, image, device, conf_thresh=0.3, num_classes=num_classes)
            except Exception as e:
                print(f"Error in inference: {e}")
                boxes, confidences, class_ids = [], [], []
            
            # GT 로드
            image_id = os.path.splitext(os.path.basename(path))[0]
            possible_label_paths = [
                f'../training/label_2/{image_id}.txt',
                f'./training/label_2/{image_id}.txt',
                f'./labels/{image_id}.txt',
                f'./label_2/{image_id}.txt',
                f'./{image_id}.txt'
            ]
            
            label_path = None
            for lp in possible_label_paths:
                if os.path.exists(lp):
                    label_path = lp
                    break
            
            gt_boxes = load_gt_boxes(label_path, KITTI_CLASSES) if label_path else []
            
            # GT 데이터 저장
            for box, class_name in gt_boxes:
                if image_id not in pytorch_annotations[class_name]:
                    pytorch_annotations[class_name][image_id] = []
                pytorch_annotations[class_name][image_id].append((box, class_name))
            
            # 예측 데이터 저장
            if len(boxes) > 0:
                for i in range(len(boxes)):
                    if i < len(class_ids) and class_ids[i] < len(KITTI_CLASSES):
                        cls_name = KITTI_CLASSES[class_ids[i]]
                        if isinstance(boxes, torch.Tensor):
                            box = boxes[i].int().tolist()
                        else:
                            box = boxes[i]
                        pytorch_detections[cls_name].append((image_id, confidences[i], box))
            
            # 상세 출력
            try:
                print_detailed_pytorch_results(image_id, boxes, confidences, class_ids, KITTI_CLASSES, gt_boxes, model_name)
            except Exception as e:
                print(f"Error in detailed output: {e}")
        
        # PyTorch mAP 계산
        pytorch_map = compute_map(pytorch_detections, pytorch_annotations, model_name, iou_thresh=0.5)
        results[model_name] = pytorch_map
    
    return results


if __name__ == "__main__":
    main()