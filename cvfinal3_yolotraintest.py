import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
from torchvision import transforms

# YOLOv4Tiny 모델 클래스 정의 (기존 코드와 동일)
class ConvBNLeaky(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBNLeaky, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x):
        return self.leaky(self.bn(self.conv(x)))

class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=1):
        super(CSPBlock, self).__init__()
        self.conv1 = ConvBNLeaky(in_channels, out_channels // 2, 1)
        self.conv2 = ConvBNLeaky(in_channels, out_channels // 2, 1)
        
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(ConvBNLeaky(out_channels // 2, out_channels // 2, 3, 1, 1))
        
        self.conv3 = ConvBNLeaky(out_channels, out_channels, 1)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        
        for block in self.blocks:
            x2 = block(x2)
        
        x = torch.cat([x1, x2], dim=1)
        return self.conv3(x)

class YOLOv4Tiny(nn.Module):
    def __init__(self, num_classes=80, anchors=None, img_size=(416, 416)):
        super(YOLOv4Tiny, self).__init__()
        self.num_classes = num_classes
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        
        if anchors is None:
            if self.img_size[1] > self.img_size[0]:  # width > height (KITTI case)
                self.anchors = [
                    [[10, 13], [16, 30], [33, 23]],  # P4
                    [[30, 61], [62, 45], [59, 119]]  # P5
                ]
            else:
                self.anchors = [
                    [[10, 14], [23, 27], [37, 58]],  # P4
                    [[81, 82], [135, 169], [344, 319]]  # P5
                ]
        else:
            self.anchors = anchors
        
        # Backbone
        self.conv1 = ConvBNLeaky(3, 32, 3, 2, 1)
        self.conv2 = ConvBNLeaky(32, 64, 3, 2, 1)
        self.csp1 = CSPBlock(64, 64, 1)
        
        self.conv3 = ConvBNLeaky(64, 128, 3, 2, 1)
        self.csp2 = CSPBlock(128, 128, 3)
        
        self.conv4 = ConvBNLeaky(128, 256, 3, 2, 1)
        self.csp3 = CSPBlock(256, 256, 3)
        
        self.conv5 = ConvBNLeaky(256, 512, 3, 2, 1)
        self.csp4 = CSPBlock(512, 512, 1)
        
        # Neck
        self.conv6 = ConvBNLeaky(512, 256, 1)
        self.conv7 = ConvBNLeaky(256, 512, 3, 1, 1)
        
        # Head
        self.conv8 = ConvBNLeaky(512, 256, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.conv9 = ConvBNLeaky(512, 256, 1)
        self.conv10 = ConvBNLeaky(256, 512, 3, 1, 1)
        
        # Output layers
        self.out1 = nn.Conv2d(512, 3 * (5 + num_classes), 1)
        self.out2 = nn.Conv2d(512, 3 * (5 + num_classes), 1)
    
    def forward(self, x):
        # Backbone
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.csp1(x)
        
        x = self.conv3(x)
        x = self.csp2(x)
        
        x = self.conv4(x)
        route1 = self.csp3(x)
        
        x = self.conv5(route1)
        x = self.csp4(x)
        
        # Neck
        x = self.conv6(x)
        x = self.conv7(x)
        
        # First output (P5)
        out1 = self.out1(x)
        
        # Second branch
        x = self.conv8(x)
        x = self.upsample(x)
        x = torch.cat([x, route1], dim=1)
        
        x = self.conv9(x)
        x = self.conv10(x)
        
        # Second output (P4)
        out2 = self.out2(x)
        
        return out1, out2

class YOLODetector:
    def __init__(self, model_path, num_classes=8, img_size=(384, 1280), device='cuda'):
        self.num_classes = num_classes
        self.img_size = img_size
        self.device = device
        
        # KITTI 클래스 이름
        self.class_names = [
            'Car', 'Van', 'Truck', 'Pedestrian', 
            'Person_sitting', 'Cyclist', 'Tram', 'Misc'
        ]
        
        # 클래스별 색상 (BGR)
        self.colors = [
            (255, 0, 0),    # Car - Red
            (0, 255, 0),    # Van - Green
            (0, 0, 255),    # Truck - Blue
            (255, 255, 0),  # Pedestrian - Cyan
            (255, 0, 255),  # Person_sitting - Magenta
            (0, 255, 255),  # Cyclist - Yellow
            (128, 0, 128),  # Tram - Purple
            (255, 165, 0)   # Misc - Orange
        ]
        
        # 모델 로드
        self.model = self.load_model(model_path)
        
    def load_model(self, model_path):
        """학습된 모델 로드"""
        print(f"Loading model from {model_path}")
        
        # 모델 생성
        model = YOLOv4Tiny(num_classes=self.num_classes, img_size=self.img_size)
        
        # 체크포인트 로드
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded successfully from epoch {checkpoint.get('epoch', 'unknown')}")
            if 'train_loss' in checkpoint:
                print(f"Training loss: {checkpoint['train_loss']:.4f}")
            if 'val_loss' in checkpoint:
                print(f"Validation loss: {checkpoint['val_loss']:.4f}")
        else:
            print(f"Warning: Model file not found at {model_path}")
            print("Using randomly initialized model")
        
        model.to(self.device)
        model.eval()
        return model
    
    def preprocess_image(self, image_path):
        """이미지 전처리"""
        # 이미지 로드
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        original_img = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_h, original_w = img.shape[:2]
        
        # 리사이즈
        img_resized = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        img_tensor = torch.from_numpy(img_resized.astype(np.float32) / 255.0)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        return img_tensor, original_img, (original_w, original_h)
    
    def postprocess_predictions(self, predictions, original_size, conf_thresh=0.5, nms_thresh=0.4):
        """예측 결과 후처리"""
        original_w, original_h = original_size
        detections = []
        
        for i, pred in enumerate(predictions):
            batch_size, _, grid_h, grid_w = pred.shape
            num_anchors = len(self.model.anchors[i])
            
            # 예측 결과 reshape
            pred = pred.view(batch_size, num_anchors, 5 + self.num_classes, grid_h, grid_w)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()
            pred = pred.view(-1, 5 + self.num_classes)
            
            # 시그모이드 적용
            pred[..., 0:2] = torch.sigmoid(pred[..., 0:2])  # x, y
            pred[..., 4] = torch.sigmoid(pred[..., 4])      # objectness
            if self.num_classes > 1:
                pred[..., 5:] = torch.sigmoid(pred[..., 5:])    # class probs
            
            # 좌표 변환
            stride_h = self.img_size[0] / grid_h
            stride_w = self.img_size[1] / grid_w
            
            for anchor_idx in range(num_anchors):
                for j in range(grid_h):
                    for k in range(grid_w):
                        idx = anchor_idx * grid_h * grid_w + j * grid_w + k
                        
                        # Objectness score
                        obj_score = pred[idx, 4].item()
                        if obj_score < conf_thresh:
                            continue
                        
                        # 좌표 계산
                        x_offset = pred[idx, 0].item()
                        y_offset = pred[idx, 1].item()
                        w_pred = pred[idx, 2].item()
                        h_pred = pred[idx, 3].item()
                        
                        # 절대 좌표 계산
                        center_x = (k + x_offset) * stride_w
                        center_y = (j + y_offset) * stride_h
                        
                        # 앵커 크기
                        anchor_w = self.model.anchors[i][anchor_idx][0]
                        anchor_h = self.model.anchors[i][anchor_idx][1]
                        
                        # 실제 크기 계산
                        width = anchor_w * np.exp(w_pred)
                        height = anchor_h * np.exp(h_pred)
                        
                        # 원본 이미지 좌표로 변환
                        x1 = (center_x - width / 2) * original_w / self.img_size[1]
                        y1 = (center_y - height / 2) * original_h / self.img_size[0]
                        x2 = (center_x + width / 2) * original_w / self.img_size[1]
                        y2 = (center_y + height / 2) * original_h / self.img_size[0]
                        
                        # 클래스 예측
                        if self.num_classes == 1:
                            class_id = 0
                            class_score = obj_score
                        else:
                            class_scores = pred[idx, 5:].cpu().numpy()
                            class_id = np.argmax(class_scores)
                            class_score = obj_score * class_scores[class_id]
                        
                        if class_score >= conf_thresh:
                            detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'score': class_score,
                                'class_id': class_id,
                                'class_name': self.class_names[class_id] if class_id < len(self.class_names) else f'Class_{class_id}'
                            })
        
        print(f"후보 박스 수 (conf 적용 후): {len(detections)}")

        # NMS 적용
        if len(detections) > 0:
            detections = self.non_max_suppression(detections, nms_thresh)
        
        return detections
    
    def non_max_suppression(self, detections, nms_thresh):
        """Non-Maximum Suppression"""
        if len(detections) == 0:
            return []
        
        # 신뢰도 기준 정렬
        detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        
        keep = []
        while len(detections) > 0:
            keep.append(detections[0])
            remaining = []
            
            for det in detections[1:]:
                iou = self.calculate_iou(keep[-1]['bbox'], det['bbox'])
                if iou < nms_thresh:
                    remaining.append(det)
            
            detections = remaining
        
        return keep
    
    def calculate_iou(self, box1, box2):
        """IoU 계산"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 교집합 영역 계산
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # 합집합 영역 계산
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def detect(self, image_path, conf_thresh=0.5, nms_thresh=0.4):
        """객체 검출 수행"""
        # 이미지 전처리
        img_tensor, original_img, original_size = self.preprocess_image(image_path)
        
        # 예측 수행
        with torch.no_grad():
            predictions = self.model(img_tensor)

        # 후처리
        detections = self.postprocess_predictions(predictions, original_size, conf_thresh, nms_thresh)
        
        return detections, original_img
    
    def visualize_detections(self, image, detections, save_path=None, show=True):
        """검출 결과 시각화"""
        img_vis = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
            score = det['score']
            class_name = det['class_name']
            class_id = det['class_id']
            
            # 바운딩 박스 그리기
            color = self.colors[class_id % len(self.colors)]
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
            
            # 라벨 그리기
            label = f"{class_name}: {score:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(img_vis, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(img_vis, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        if save_path:
            cv2.imwrite(save_path, img_vis)
            print(f"Result saved to {save_path}")
        
        if show:
            plt.figure(figsize=(15, 10))
            plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
            plt.title(f'YOLOv4 Tiny Detection Results ({len(detections)} objects)')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        return img_vis

def test_single_image(model_path, image_path, conf_thresh=0.5, nms_thresh=0.4):
    """단일 이미지 테스트"""
    print(f"Testing image: {image_path}")
    
    # 검출기 초기화
    detector = YOLODetector(
        model_path=model_path,
        num_classes=8,  # KITTI 클래스 수
        img_size=(384, 1280),  # 학습 시 사용한 이미지 크기
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 검출 수행
    detections, original_img = detector.detect(image_path, conf_thresh, nms_thresh)
    
    # 결과 출력
    print(f"Found {len(detections)} objects:")
    for i, det in enumerate(detections):
        print(f"  {i+1}. {det['class_name']}: {det['score']:.3f} "
              f"[{det['bbox'][0]:.1f}, {det['bbox'][1]:.1f}, {det['bbox'][2]:.1f}, {det['bbox'][3]:.1f}]")
    
    # 시각화
    result_img = detector.visualize_detections(
        original_img, detections, 
        save_path=f"detection_result_{os.path.basename(image_path)}"
    )
    
    return detections, result_img

def test_multiple_images(model_path, image_dir, output_dir="results", conf_thresh=0.5, nms_thresh=0.4):
    """여러 이미지 테스트"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 검출기 초기화
    detector = YOLODetector(
        model_path=model_path,
        num_classes=8,
        img_size=(384, 1280),
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    print(f"Testing {len(image_files)} images...")
    
    all_results = []
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        print(f"\nProcessing: {img_file}")
        
        try:
            # 검출 수행
            detections, original_img = detector.detect(img_path, conf_thresh, nms_thresh)
            
            # 결과 저장
            output_path = os.path.join(output_dir, f"result_{img_file}")
            detector.visualize_detections(
                original_img, detections, 
                save_path=output_path, show=False
            )
            
            # 통계 저장
            result = {
                'image': img_file,
                'detections': len(detections),
                'objects': [det['class_name'] for det in detections]
            }
            all_results.append(result)
            
            print(f"  Found {len(detections)} objects: {[det['class_name'] for det in detections]}")
            
        except Exception as e:
            print(f"  Error processing {img_file}: {str(e)}")
    
    # 전체 결과 요약
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    total_detections = sum(r['detections'] for r in all_results)
    print(f"Total images processed: {len(all_results)}")
    print(f"Total objects detected: {total_detections}")
    print(f"Average objects per image: {total_detections/len(all_results):.1f}")
    
    # 클래스별 통계
    class_counts = {}
    for result in all_results:
        for obj in result['objects']:
            class_counts[obj] = class_counts.get(obj, 0) + 1
    
    print("\nDetected classes:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {count}")
    
    return all_results

# 사용 예시
if __name__ == "__main__":
    # 설정
    MODEL_PATH = "yolov4_tiny_best.pth"  # 또는 "yolov4_tiny_epoch_30.pth"
    
    # 여러 이미지 테스트 (선택사항)
    TEST_IMAGE_DIR = "../data_object_image_2/training/image_2"  # 테스트할 이미지 디렉토리
    
    if os.path.exists(TEST_IMAGE_DIR):
        print("\n=== Multiple Images Test ===")
        # 처음 10개 이미지만 테스트
        test_files = [f for f in os.listdir(TEST_IMAGE_DIR) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))][:10]
        
        if test_files:
            # 임시 디렉토리에 테스트 이미지 복사
            temp_dir = "temp_test_images"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            
            import shutil
            for f in test_files:
                shutil.copy(os.path.join(TEST_IMAGE_DIR, f), temp_dir)
            
            results = test_multiple_images(
                model_path=MODEL_PATH,
                image_dir=temp_dir,
                output_dir="detection_results",
                conf_thresh=0.25,
                nms_thresh=0.5
            )