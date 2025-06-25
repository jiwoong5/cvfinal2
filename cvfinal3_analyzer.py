import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
import glob
from collections import defaultdict
import pandas as pd
import seaborn as sns

class KITTIBoxAnalyzer:
    """KITTI 데이터셋의 GT 박스와 예측 박스 크기 분석 도구"""
    
    def __init__(self, kitti_label_dir, kitti_image_dir, detector=None):
        self.label_dir = kitti_label_dir
        self.image_dir = kitti_image_dir
        self.detector = detector
        
        # KITTI 클래스 매핑
        self.kitti_class_map = {
            'Car': 0, 'Van': 1, 'Truck': 2, 'Pedestrian': 3,
            'Person_sitting': 4, 'Cyclist': 5, 'Tram': 6, 
            'Misc': 7, 'DontCare': -1
        }
        
        self.class_names = [
            'Car', 'Van', 'Truck', 'Pedestrian', 
            'Person_sitting', 'Cyclist', 'Tram', 'Misc'
        ]
    
    def parse_kitti_label(self, label_path):
        """KITTI 라벨 파일 파싱"""
        boxes = []
        
        if not os.path.exists(label_path):
            return boxes
            
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
                
            class_name = parts[0]
            if class_name == 'DontCare':
                continue
                
            # 2D 바운딩 박스 좌표 (left, top, right, bottom)
            left = float(parts[4])
            top = float(parts[5])
            right = float(parts[6])
            bottom = float(parts[7])
            
            width = right - left
            height = bottom - top
            
            if width > 0 and height > 0:  # 유효한 박스만
                box_info = {
                    'class_name': class_name,
                    'class_id': self.kitti_class_map.get(class_name, -1),
                    'bbox': [left, top, right, bottom],
                    'width': width,
                    'height': height,
                    'area': width * height,
                    'aspect_ratio': width / height
                }
                boxes.append(box_info)
        
        return boxes
    
    def analyze_gt_boxes(self, num_images=10):
        """GT 박스 크기 분석"""
        print(f"Analyzing GT boxes from {num_images} images...")
        
        # 라벨 파일 목록 가져오기
        label_files = sorted(glob.glob(os.path.join(self.label_dir, "*.txt")))
        if len(label_files) == 0:
            print(f"No label files found in {self.label_dir}")
            return None
            
        # 처음 num_images개 파일만 분석
        label_files = label_files[:num_images]
        
        all_gt_boxes = []
        class_stats = defaultdict(list)
        
        for label_file in label_files:
            boxes = self.parse_kitti_label(label_file)
            all_gt_boxes.extend(boxes)
            
            # 클래스별 통계 수집
            for box in boxes:
                if box['class_id'] >= 0:  # DontCare 제외
                    class_stats[box['class_name']].append({
                        'width': box['width'],
                        'height': box['height'],
                        'area': box['area'],
                        'aspect_ratio': box['aspect_ratio']
                    })
        
        print(f"Total GT boxes analyzed: {len(all_gt_boxes)}")
        
        # 전체 통계
        gt_stats = self._calculate_box_statistics(all_gt_boxes)
        
        # 클래스별 통계
        class_detailed_stats = {}
        for class_name, boxes in class_stats.items():
            class_detailed_stats[class_name] = self._calculate_box_statistics(boxes)
        
        return {
            'total_boxes': len(all_gt_boxes),
            'overall_stats': gt_stats,
            'class_stats': class_detailed_stats,
            'raw_boxes': all_gt_boxes
        }
    
    def analyze_prediction_boxes(self, num_images=10, conf_thresh=0.25):
        """예측 박스 크기 분석"""
        if self.detector is None:
            print("No detector provided for prediction analysis")
            return None
            
        print(f"Analyzing prediction boxes from {num_images} images...")
        
        # 이미지 파일 목록 가져오기
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(glob.glob(os.path.join(self.image_dir, ext)))
        
        image_files = sorted(image_files)[:num_images]
        
        all_pred_boxes = []
        class_stats = defaultdict(list)
        
        for img_file in image_files:
            print(f"Processing: {os.path.basename(img_file)}")
            
            try:
                # 예측 수행
                detections, _ = self.detector.detect(img_file, conf_thresh=conf_thresh)
                
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    width = x2 - x1
                    height = y2 - y1

                    box_info = {
                        'class_name': det['class_name'],
                        'class_id': det['class_id'],
                        'bbox': det['bbox'],
                        'width': width,
                        'height': height,
                        'area': width * height,
                        'aspect_ratio': width / height if height > 0 else 0,
                        'confidence': det['score']
                    }
                    
                    all_pred_boxes.append(box_info)
                    class_stats[det['class_name']].append({
                        'width': width,
                        'height': height,
                        'area': width * height,
                        'aspect_ratio': width / height if height > 0 else 0
                    })
                    
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
        
        print(f"Total prediction boxes: {len(all_pred_boxes)}")
        
        # 전체 통계
        pred_stats = self._calculate_box_statistics(all_pred_boxes)
        
        # 클래스별 통계
        class_detailed_stats = {}
        for class_name, boxes in class_stats.items():
            class_detailed_stats[class_name] = self._calculate_box_statistics(boxes)
        
        return {
            'total_boxes': len(all_pred_boxes),
            'overall_stats': pred_stats,
            'class_stats': class_detailed_stats,
            'raw_boxes': all_pred_boxes
        }
    
    def _calculate_box_statistics(self, boxes):
        """박스 통계 계산"""
        if not boxes:
            return {}
            
        widths = [box['width'] for box in boxes]
        heights = [box['height'] for box in boxes]
        areas = [box['area'] for box in boxes]
        aspect_ratios = [box['aspect_ratio'] for box in boxes if box['aspect_ratio'] > 0]
        
        stats = {
            'count': len(boxes),
            'width': {
                'mean': np.mean(widths),
                'std': np.std(widths),
                'min': np.min(widths),
                'max': np.max(widths),
                'median': np.median(widths)
            },
            'height': {
                'mean': np.mean(heights),
                'std': np.std(heights),
                'min': np.min(heights),
                'max': np.max(heights),
                'median': np.median(heights)
            },
            'area': {
                'mean': np.mean(areas),
                'std': np.std(areas),
                'min': np.min(areas),
                'max': np.max(areas),
                'median': np.median(areas)
            },
            'aspect_ratio': {
                'mean': np.mean(aspect_ratios) if aspect_ratios else 0,
                'std': np.std(aspect_ratios) if aspect_ratios else 0,
                'min': np.min(aspect_ratios) if aspect_ratios else 0,
                'max': np.max(aspect_ratios) if aspect_ratios else 0,
                'median': np.median(aspect_ratios) if aspect_ratios else 0
            }
        }
        
        return stats
    
    def compare_gt_vs_prediction(self, gt_analysis, pred_analysis):
        """GT와 예측 박스 비교"""
        print("\n" + "="*80)
        print("GT vs PREDICTION COMPARISON")
        print("="*80)
        
        # 전체 비교
        print("\n[OVERALL COMPARISON]")
        print(f"GT boxes: {gt_analysis['total_boxes']}")
        print(f"Prediction boxes: {pred_analysis['total_boxes']}")
        
        gt_overall = gt_analysis['overall_stats']
        pred_overall = pred_analysis['overall_stats']
        
        print(f"\nWidth comparison:")
        print(f"  GT:     mean={gt_overall['width']['mean']:.1f}, std={gt_overall['width']['std']:.1f}")
        print(f"  Pred:   mean={pred_overall['width']['mean']:.1f}, std={pred_overall['width']['std']:.1f}")
        print(f"  Ratio:  {pred_overall['width']['mean']/gt_overall['width']['mean']:.2f}")
        
        print(f"\nHeight comparison:")
        print(f"  GT:     mean={gt_overall['height']['mean']:.1f}, std={gt_overall['height']['std']:.1f}")
        print(f"  Pred:   mean={pred_overall['height']['mean']:.1f}, std={pred_overall['height']['std']:.1f}")
        print(f"  Ratio:  {pred_overall['height']['mean']/gt_overall['height']['mean']:.2f}")
        
        print(f"\nArea comparison:")
        print(f"  GT:     mean={gt_overall['area']['mean']:.0f}, std={gt_overall['area']['std']:.0f}")
        print(f"  Pred:   mean={pred_overall['area']['mean']:.0f}, std={pred_overall['area']['std']:.0f}")
        print(f"  Ratio:  {pred_overall['area']['mean']/gt_overall['area']['mean']:.2f}")
        
        # 클래스별 비교
        print(f"\n[CLASS-WISE COMPARISON]")
        common_classes = set(gt_analysis['class_stats'].keys()) & set(pred_analysis['class_stats'].keys())
        
        for class_name in sorted(common_classes):
            gt_class = gt_analysis['class_stats'][class_name]
            pred_class = pred_analysis['class_stats'][class_name]
            
            print(f"\n{class_name}:")
            print(f"  Count - GT: {gt_class['count']}, Pred: {pred_class['count']}")
            print(f"  Width - GT: {gt_class['width']['mean']:.1f}±{gt_class['width']['std']:.1f}, "
                  f"Pred: {pred_class['width']['mean']:.1f}±{pred_class['width']['std']:.1f}")
            print(f"  Height - GT: {gt_class['height']['mean']:.1f}±{gt_class['height']['std']:.1f}, "
                  f"Pred: {pred_class['height']['mean']:.1f}±{pred_class['height']['std']:.1f}")
    
    def visualize_box_distributions(self, gt_analysis, pred_analysis, save_path="box_analysis.png"):
        """박스 크기 분포 시각화"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('GT vs Prediction Box Size Distributions', fontsize=16, fontweight='bold')
        
        # 데이터 준비
        gt_boxes = gt_analysis['raw_boxes']
        pred_boxes = pred_analysis['raw_boxes']
        
        gt_widths = [box['width'] for box in gt_boxes]
        gt_heights = [box['height'] for box in gt_boxes]
        gt_areas = [box['area'] for box in gt_boxes]
        
        pred_widths = [box['width'] for box in pred_boxes]
        pred_heights = [box['height'] for box in pred_boxes]
        pred_areas = [box['area'] for box in pred_boxes]
        
        # Width 분포
        axes[0, 0].hist(gt_widths, bins=30, alpha=0.7, label='GT', color='blue', density=True)
        axes[0, 0].hist(pred_widths, bins=30, alpha=0.7, label='Prediction', color='red', density=True)
        axes[0, 0].set_title('Width Distribution')
        axes[0, 0].set_xlabel('Width (pixels)')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Height 분포
        axes[0, 1].hist(gt_heights, bins=30, alpha=0.7, label='GT', color='blue', density=True)
        axes[0, 1].hist(pred_heights, bins=30, alpha=0.7, label='Prediction', color='red', density=True)
        axes[0, 1].set_title('Height Distribution')
        axes[0, 1].set_xlabel('Height (pixels)')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Area 분포
        axes[0, 2].hist(gt_areas, bins=30, alpha=0.7, label='GT', color='blue', density=True)
        axes[0, 2].hist(pred_areas, bins=30, alpha=0.7, label='Prediction', color='red', density=True)
        axes[0, 2].set_title('Area Distribution')
        axes[0, 2].set_xlabel('Area (pixels²)')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Width vs Height 산점도
        axes[1, 0].scatter(gt_widths, gt_heights, alpha=0.6, label='GT', color='blue', s=20)
        axes[1, 0].scatter(pred_widths, pred_heights, alpha=0.6, label='Prediction', color='red', s=20)
        axes[1, 0].set_title('Width vs Height')
        axes[1, 0].set_xlabel('Width (pixels)')
        axes[1, 0].set_ylabel('Height (pixels)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 클래스별 평균 크기 비교
        common_classes = set(gt_analysis['class_stats'].keys()) & set(pred_analysis['class_stats'].keys())
        if common_classes:
            classes = sorted(list(common_classes))
            gt_mean_widths = [gt_analysis['class_stats'][cls]['width']['mean'] for cls in classes]
            pred_mean_widths = [pred_analysis['class_stats'][cls]['width']['mean'] for cls in classes]
            
            x = np.arange(len(classes))
            width = 0.35
            
            axes[1, 1].bar(x - width/2, gt_mean_widths, width, label='GT', color='blue', alpha=0.7)
            axes[1, 1].bar(x + width/2, pred_mean_widths, width, label='Prediction', color='red', alpha=0.7)
            axes[1, 1].set_title('Average Width by Class')
            axes[1, 1].set_xlabel('Class')
            axes[1, 1].set_ylabel('Average Width (pixels)')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(classes, rotation=45)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 박스 크기 vs 현재 앵커 비교
        current_anchors = [
            [40, 32], [228, 122], [107, 83],  # P4
            [395, 187], [300, 193], [354, 348]  # P5
        ]
        
        # 앵커를 이미지 크기에 맞게 스케일링 (stride 고려)
        img_size = (384, 1280)
        stride_p4 = (img_size[0] // 24, img_size[1] // 80)  # 대략적인 P4 stride
        stride_p5 = (img_size[0] // 12, img_size[1] // 40)  # 대략적인 P5 stride
        
        scaled_anchors = []
        for i, anchor in enumerate(current_anchors):
            if i < 3:  # P4 앵커
                scaled_w = anchor[0] * stride_p4[1] / img_size[1] * 1280  # 원본 이미지 크기로 변환
                scaled_h = anchor[1] * stride_p4[0] / img_size[0] * 384
            else:  # P5 앵커
                scaled_w = anchor[0] * stride_p5[1] / img_size[1] * 1280
                scaled_h = anchor[1] * stride_p5[0] / img_size[0] * 384
            scaled_anchors.append([scaled_w, scaled_h])
        
        axes[1, 2].scatter(gt_widths, gt_heights, alpha=0.6, label='GT boxes', color='blue', s=20)
        for i, anchor in enumerate(scaled_anchors):
            axes[1, 2].scatter(anchor[0], anchor[1], marker='x', s=100, 
                             color='red' if i < 3 else 'orange', 
                             label=f'Anchor P{"4" if i < 3 else "5"}-{i%3+1}' if i in [0, 3] else "")
        
        axes[1, 2].set_title('GT Boxes vs Current Anchors')
        axes[1, 2].set_xlabel('Width (pixels)')
        axes[1, 2].set_ylabel('Height (pixels)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
        plt.show()
    
    def suggest_better_anchors(self, gt_analysis, num_anchors=6):
        """GT 박스를 기반으로 더 나은 앵커 제안"""
        print(f"\n[ANCHOR OPTIMIZATION SUGGESTIONS]")
        
        gt_boxes = gt_analysis['raw_boxes']
        if len(gt_boxes) < num_anchors:
            print(f"Not enough GT boxes ({len(gt_boxes)}) for {num_anchors} anchors")
            return
        
        # GT 박스의 width, height 추출
        box_sizes = np.array([[box['width'], box['height']] for box in gt_boxes])
        
        # K-means 클러스터링으로 앵커 생성
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=num_anchors, random_state=42, n_init=10)
        kmeans.fit(box_sizes)
        
        suggested_anchors = kmeans.cluster_centers_
        suggested_anchors = sorted(suggested_anchors, key=lambda x: x[0] * x[1])  # 면적 기준 정렬
        
        print(f"\nCurrent anchors (scaled to image size ~1280x384):")
        current_anchors = [
            [10, 13], [16, 30], [33, 23],  # P4
            [30, 61], [62, 45], [59, 119]  # P5
        ]
        for i, anchor in enumerate(current_anchors):
            layer = "P4" if i < 3 else "P5"
            print(f"  {layer}-{i%3+1}: [{anchor[0]:3.0f}, {anchor[1]:3.0f}]")
        
        print(f"\nSuggested anchors based on GT analysis:")
        for i, anchor in enumerate(suggested_anchors):
            layer = "P4" if i < 3 else "P5"
            print(f"  {layer}-{i%3+1}: [{anchor[0]:3.0f}, {anchor[1]:3.0f}]")
        
        # 앵커 품질 평가 (IoU 기반)
        current_ious = self._calculate_anchor_ious(box_sizes, current_anchors)
        suggested_ious = self._calculate_anchor_ious(box_sizes, suggested_anchors)
        
        print(f"\nAnchor quality (average IoU with GT boxes):")
        print(f"  Current anchors:   {np.mean(current_ious):.3f}")
        print(f"  Suggested anchors: {np.mean(suggested_ious):.3f}")
        print(f"  Improvement:       {np.mean(suggested_ious) - np.mean(current_ious):+.3f}")
        
        return suggested_anchors
    
    def _calculate_anchor_ious(self, gt_boxes, anchors):
        """앵커와 GT 박스 간의 IoU 계산"""
        ious = []
        
        for gt_box in gt_boxes:
            gt_w, gt_h = gt_box[0], gt_box[1]
            best_iou = 0
            
            for anchor in anchors:
                anchor_w, anchor_h = anchor[0], anchor[1]
                
                # IoU 계산 (중심점이 같다고 가정)
                intersection = min(gt_w, anchor_w) * min(gt_h, anchor_h)
                union = gt_w * gt_h + anchor_w * anchor_h - intersection
                iou = intersection / union if union > 0 else 0
                
                best_iou = max(best_iou, iou)
            
            ious.append(best_iou)
        
        return ious

def run_complete_analysis(model_path, kitti_label_dir, kitti_image_dir, num_images=10):
    """완전한 분석 실행"""
    from cvfinal3_yolotraintest import YOLODetector  # 실제 detector import 경로로 수정
    
    # 검출기 초기화
    detector = YOLODetector(
        model_path=model_path,
        num_classes=8,
        img_size=(384, 1280),
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 분석기 초기화
    analyzer = KITTIBoxAnalyzer(kitti_label_dir, kitti_image_dir, detector)
    
    print("Starting comprehensive box size analysis...")
    
    # GT 분석
    print("\n1. Analyzing Ground Truth boxes...")
    gt_analysis = analyzer.analyze_gt_boxes(num_images)
    
    # 예측 분석
    print("\n2. Analyzing Prediction boxes...")
    pred_analysis = analyzer.analyze_prediction_boxes(num_images, conf_thresh=0.25)
    
    # 비교 분석
    print("\n3. Comparing GT vs Predictions...")
    analyzer.compare_gt_vs_prediction(gt_analysis, pred_analysis)
    
    # 시각화
    print("\n4. Creating visualizations...")
    analyzer.visualize_box_distributions(gt_analysis, pred_analysis)
    
    # 앵커 최적화 제안
    print("\n5. Suggesting better anchors...")
    analyzer.suggest_better_anchors(gt_analysis)
    
    return gt_analysis, pred_analysis

# 사용 예시
if __name__ == "__main__":
    # 설정
    MODEL_PATH = "./yolov4_tiny_epoch_30.pth"
    KITTI_LABEL_DIR = "../training/label_2"
    KITTI_IMAGE_DIR = "../data_object_image_2/training/image_2"
    
    # 완전한 분석 실행
    gt_analysis, pred_analysis = run_complete_analysis(
        model_path=MODEL_PATH,
        kitti_label_dir=KITTI_LABEL_DIR,
        kitti_image_dir=KITTI_IMAGE_DIR,
        num_images=10
    )