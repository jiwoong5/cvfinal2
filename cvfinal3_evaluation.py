import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import os
import random
from PIL import Image
import torchvision.transforms as transforms
from cvfinal3_yolotrain import ConvBNLeaky, CSPBlock, YOLOv4Tiny

# 기존 모델 클래스들을 import (위의 코드에서 정의된 클래스들)
# ConvBNLeaky, CSPBlock, YOLOv4Tiny 클래스들이 이미 정의되어 있다고 가정

class YOLOInference:
    def __init__(self, model_path, num_classes=8, img_size=(384, 1280), conf_threshold=0.5, nms_threshold=0.4):
        self.num_classes = num_classes
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        # KITTI 클래스 이름
        self.class_names = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
        
        # 클래스별 색상 정의
        self.colors = [
            (255, 0, 0),     # Car - Red
            (0, 255, 0),     # Van - Green
            (0, 0, 255),     # Truck - Blue
            (255, 255, 0),   # Pedestrian - Yellow
            (255, 0, 255),   # Person_sitting - Magenta
            (0, 255, 255),   # Cyclist - Cyan
            (128, 0, 255),   # Tram - Purple
            (255, 128, 0)    # Misc - Orange
        ]
        
        # 모델 로드
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        
    def load_model(self, model_path):
        """학습된 모델 가중치 로드"""
        model = YOLOv4Tiny(num_classes=self.num_classes, img_size=self.img_size)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
            
        model.to(self.device)
        model.eval()
        return model
        
    def preprocess_image(self, image_path):
        """이미지 전처리"""
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_shape = img.shape[:2]  # (height, width)
        
        # 리사이즈
        img_resized = cv2.resize(img, (self.img_size[1], self.img_size[0]))  # (width, height)
        img_tensor = torch.from_numpy(img_resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        
        return img, img_tensor.to(self.device), original_shape
        
    def postprocess_predictions(self, predictions, original_shape):
        detections = []

        for i, pred in enumerate(predictions):
            # pred가 4차원일 경우
            batch_size, channels, grid_h, grid_w = pred.shape

            num_anchors = len(self.model.anchors[i])
            num_outputs = 5 + self.num_classes  # bbox 4 + obj + classes

            # pred reshape 및 permute 해서 5차원 텐서로 변경
            pred = pred.view(batch_size, num_anchors, num_outputs, grid_h, grid_w)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()  # (B, A, H, W, O)

            # 이제 pred는 5차원이므로 unpack 가능
            batch_size, num_anchors, grid_h, grid_w, num_outputs = pred.shape

            pred_clone = pred.clone()

            # 시그모이드 적용
            pred_clone[..., 0:2] = torch.sigmoid(pred_clone[..., 0:2])  # x, y
            pred_clone[..., 4] = torch.sigmoid(pred_clone[..., 4])      # objectness
            if self.num_classes > 1:
                pred_clone[..., 5:] = torch.sigmoid(pred_clone[..., 5:])  # classes

            for b in range(batch_size):
                for a in range(num_anchors):
                    for j in range(grid_h):
                        for k in range(grid_w):
                            objectness = pred_clone[b, a, j, k, 4].item()
                            if objectness < self.conf_threshold:
                                continue

                            x_offset = pred_clone[b, a, j, k, 0].item()
                            y_offset = pred_clone[b, a, j, k, 1].item()
                            w_exp = pred_clone[b, a, j, k, 2].item()
                            h_exp = pred_clone[b, a, j, k, 3].item()

                            anchor_w = self.model.anchors[i][a][0] / (self.img_size[1] / grid_w)
                            anchor_h = self.model.anchors[i][a][1] / (self.img_size[0] / grid_h)

                            center_x = (k + x_offset) / grid_w
                            center_y = (j + y_offset) / grid_h
                            width = (anchor_w * np.exp(w_exp)) / grid_w
                            height = (anchor_h * np.exp(h_exp)) / grid_h

                            if self.num_classes > 1:
                                class_probs = pred_clone[b, a, j, k, 5:].cpu().numpy()
                                class_id = np.argmax(class_probs)
                                class_conf = class_probs[class_id]
                            else:
                                class_id = 0
                                class_conf = 1.0

                            final_conf = objectness * class_conf

                            if final_conf >= self.conf_threshold:
                                x1 = int((center_x - width / 2) * original_shape[1])
                                y1 = int((center_y - height / 2) * original_shape[0])
                                x2 = int((center_x + width / 2) * original_shape[1])
                                y2 = int((center_y + height / 2) * original_shape[0])

                                detections.append({
                                    'bbox': [x1, y1, x2, y2],
                                    'confidence': final_conf,
                                    'class_id': class_id,
                                    'class_name': self.class_names[class_id]
                                })

        # NMS 적용 등 이후 로직 유지
        detections = self.apply_nms(detections)
        return detections

        
    def apply_nms(self, detections):
        """Non-Maximum Suppression 적용"""
        if len(detections) == 0:
            return detections
            
        # 신뢰도로 정렬
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        final_detections = []
        
        while detections:
            # 가장 높은 신뢰도의 detection을 선택
            best = detections.pop(0)
            final_detections.append(best)
            
            # 나머지 detection들과 IoU 계산
            remaining = []
            for det in detections:
                if self.calculate_iou(best['bbox'], det['bbox']) < self.nms_threshold:
                    remaining.append(det)
            
            detections = remaining
            
        return final_detections
        
    def calculate_iou(self, box1, box2):
        """IoU 계산"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 교집합 영역
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0
            
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # 합집합 영역
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
        
    def predict(self, image_path):
        """이미지에 대한 예측 수행"""
        original_img, img_tensor, original_shape = self.preprocess_image(image_path)
        
        with torch.no_grad():
            predictions = self.model(img_tensor)
            
        detections = self.postprocess_predictions(predictions, original_shape)
        return original_img, detections

def load_gt_annotations(label_path, img_width, img_height):
    """GT 어노테이션 로드 (KITTI 형식)"""
    kitti_classes = {
        'Car': 0, 'Van': 1, 'Truck': 2, 'Pedestrian': 3, 
        'Person_sitting': 4, 'Cyclist': 5, 'Tram': 6, 'Misc': 7
    }
    class_names = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
    
    annotations = []
    
    if not os.path.exists(label_path):
        return annotations
        
    with open(label_path, 'r') as f:
        for line in f.readlines():
            data = line.strip().split()
            if len(data) < 15:
                continue
                
            class_name = data[0]
            if class_name not in kitti_classes:
                continue
                
            # 바운딩 박스 좌표
            bbox_left = float(data[4])
            bbox_top = float(data[5])
            bbox_right = float(data[6])
            bbox_bottom = float(data[7])
            
            # 유효성 검사
            if bbox_right <= bbox_left or bbox_bottom <= bbox_top:
                continue
            if bbox_left < 0 or bbox_top < 0 or bbox_right > img_width or bbox_bottom > img_height:
                continue
                
            annotations.append({
                'bbox': [int(bbox_left), int(bbox_top), int(bbox_right), int(bbox_bottom)],
                'class_id': kitti_classes[class_name],
                'class_name': class_name
            })
    
    return annotations

def draw_boxes(image, detections, colors, title="", is_gt=False):
    """이미지에 바운딩 박스 그리기"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.imshow(image)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        class_id = det['class_id']
        class_name = det['class_name']
        
        # 색상 설정 (RGB를 0-1 범위로 변환)
        color = [c/255.0 for c in colors[class_id]]
        
        # 바운딩 박스 그리기
        rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                        linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # 라벨 텍스트
        if is_gt:
            label = f"{class_name}"
        else:
            confidence = det.get('confidence', 1.0)
            label = f"{class_name}: {confidence:.2f}"
            
        # 텍스트 배경
        ax.text(x1, y1-5, label, fontsize=10, color='white', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8))
    
    return fig

def qualitative_evaluation(model_path, img_dir, label_dir, num_samples=5, 
                         conf_threshold=0.3, nms_threshold=0.4):
    """정성적 평가 수행"""
    
    # 추론 객체 생성
    inference = YOLOInference(model_path, conf_threshold=conf_threshold, 
                             nms_threshold=nms_threshold)
    
    # 이미지 파일 목록 가져오기
    img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    # 랜덤하게 샘플 선택
    selected_files = random.sample(img_files, min(num_samples, len(img_files)))
    
    print(f"Selected {len(selected_files)} images for evaluation:")
    for i, file in enumerate(selected_files):
        print(f"{i+1}. {file}")
    
    # 결과를 저장할 figure 생성
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, 2)
    
    plt.suptitle('Qualitative Evaluation: GT vs Predictions', fontsize=16, fontweight='bold')
    
    for i, img_file in enumerate(selected_files):
        img_path = os.path.join(img_dir, img_file)
        label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
        label_path = os.path.join(label_dir, label_file)
        
        print(f"\nProcessing {img_file}...")
        
        # 이미지 로드
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_height, img_width = image.shape[:2]
        
        # GT 로드
        gt_annotations = load_gt_annotations(label_path, img_width, img_height)
        print(f"  GT objects: {len(gt_annotations)}")
        
        # 예측 수행
        _, predictions = inference.predict(img_path)
        print(f"  Predicted objects: {len(predictions)}")
        
        # GT 시각화 (상단)
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f'GT - {img_file}', fontsize=10)
        axes[i, 0].axis('off')
        
        for gt in gt_annotations:
            x1, y1, x2, y2 = gt['bbox']
            class_id = gt['class_id']
            color = [c/255.0 for c in inference.colors[class_id]]

            rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                            linewidth=2, edgecolor=color, facecolor='none')
            axes[i, 0].add_patch(rect)
            axes[i, 0].text(x1, y1 - 5, gt['class_name'], fontsize=8, color='white',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8))
        
        # 예측 시각화 (하단)
        axes[i, 1].imshow(image)
        axes[i, 1].set_title(f'Prediction - {img_file}', fontsize=10)
        axes[i, 1].axis('off')
        
        for pred in predictions:
            x1, y1, x2, y2 = pred['bbox']
            class_id = pred['class_id']
            color = [c/255.0 for c in inference.colors[class_id]]

            rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                            linewidth=2, edgecolor=color, facecolor='none')
            axes[i, 1].add_patch(rect)

            label = f"{pred['class_name']}: {pred['confidence']:.2f}"
            axes[i, 1].text(x1, y1 - 5, label, fontsize=8, color='white',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8))

    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # 범례 추가
    legend_elements = []
    for i, class_name in enumerate(inference.class_names):
        color = [c/255.0 for c in inference.colors[i]]
        legend_elements.append(plt.Line2D([0], [0], color=color, lw=4, label=class_name))
    
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), 
              ncol=len(inference.class_names), fontsize=10)
    
    plt.savefig("./output/3_comparison", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 개별 상세 결과 출력
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)
    
    for i, img_file in enumerate(selected_files):
        img_path = os.path.join(img_dir, img_file)
        label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
        label_path = os.path.join(label_dir, label_file)
        
        # GT 및 예측 재로드
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_height, img_width = image.shape[:2]
        
        gt_annotations = load_gt_annotations(label_path, img_width, img_height)
        _, predictions = inference.predict(img_path)
        
        print(f"\nImage {i+1}: {img_file}")
        print(f"Image size: {img_width} x {img_height}")
        print(f"Ground Truth ({len(gt_annotations)} objects):")
        for j, gt in enumerate(gt_annotations):
            print(f"  {j+1}. {gt['class_name']} - {gt['bbox']}")
            
        print(f"Predictions ({len(predictions)} objects):")
        for j, pred in enumerate(predictions):
            print(f"  {j+1}. {pred['class_name']} ({pred['confidence']:.3f}) - {pred['bbox']}")

def calculate_iou(box1, box2):
    """IoU 계산 (이미 정의되어 있으니 재사용)"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

def quantitative_evaluation(model_path, img_dir, label_dir, iou_threshold=0.5):
    """정량적 평가 (Precision, Recall) 간단 구현"""
    inference = YOLOInference(model_path)
    img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    total_gt = 0
    total_pred = 0
    total_tp = 0

    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        label_file = img_file.rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)

        # 이미지 불러오기 (크기 필요)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]

        gt_boxes = load_gt_annotations(label_path, img_w, img_h)
        preds = inference.predict(img_path)[1]

        total_gt += len(gt_boxes)
        total_pred += len(preds)

        matched_gt = set()
        matched_pred = set()

        # GT와 예측 매칭 (클래스가 같고 IoU > threshold)
        for pi, pred in enumerate(preds):
            pred_box = pred['bbox']
            pred_cls = pred['class_id']
            for gi, gt in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                if pred_cls != gt['class_id']:
                    continue
                iou = calculate_iou(pred_box, gt['bbox'])
                if iou >= iou_threshold:
                    total_tp += 1
                    matched_gt.add(gi)
                    matched_pred.add(pi)
                    break

    precision = total_tp / total_pred if total_pred > 0 else 0
    recall = total_tp / total_gt if total_gt > 0 else 0

    print(f"\nQuantitative Evaluation Results:")
    print(f"Total GT objects: {total_gt}")
    print(f"Total Predicted objects: {total_pred}")
    print(f"True Positives: {total_tp}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

# 사용 예시
if __name__ == "__main__":
    # 설정
    MODEL_PATH = "./yolov4_tiny_epoch_30.pth"  # 학습된 모델 경로
    IMG_DIR = "../data_object_image_2/training/image_2"  # 이미지 디렉토리
    LABEL_DIR = "../training/label_2"  # 라벨 디렉토리
    NUM_SAMPLES = 5  # 평가할 이미지 수
    CONF_THRESHOLD = 0.3  # 신뢰도 임계값
    NMS_THRESHOLD = 0.4   # NMS 임계값
    
    # 정성적 평가 실행
    qualitative_evaluation(
        model_path=MODEL_PATH,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        num_samples=NUM_SAMPLES,
        conf_threshold=CONF_THRESHOLD,
        nms_threshold=NMS_THRESHOLD
    )

    # 정량적 평가
    quantitative_evaluation(
        model_path=MODEL_PATH,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        iou_threshold=0.5
    )