import os
import numpy as np
from sklearn.cluster import KMeans

def load_kitti_bboxes(label_dir):
    """
    KITTI 라벨에서 바운딩 박스 너비, 높이 (픽셀 단위) 리스트 추출
    """
    whs = []
    for label_file in os.listdir(label_dir):
        if not label_file.endswith('.txt'):
            continue
        label_path = os.path.join(label_dir, label_file)
        with open(label_path, 'r') as f:
            for line in f.readlines():
                data = line.strip().split()
                if len(data) < 15:
                    continue
                
                # bbox 좌표는 인덱스 4~7
                try:
                    bbox_left = float(data[4])
                    bbox_top = float(data[5])
                    bbox_right = float(data[6])
                    bbox_bottom = float(data[7])
                except:
                    continue
                
                w = bbox_right - bbox_left
                h = bbox_bottom - bbox_top
                if w > 0 and h > 0:
                    whs.append([w, h])
    return np.array(whs)


def split_scales(whs, img_width, img_height, strides):
    """
    개선된 스케일 분리: bbox 크기를 기준으로 적절한 스케일에 할당
    strides: [16, 32] (P4, P5의 stride)
    """
    scales = [[] for _ in strides]
    
    for w, h in whs:
        # bbox 면적 기준으로 스케일 결정
        bbox_area = w * h
        img_area = img_width * img_height
        relative_area = bbox_area / img_area
        
        # 작은 객체는 P4(stride=16), 큰 객체는 P5(stride=32)
        if relative_area < 0.1:  # 임계값은 데이터셋에 따라 조정
            scales[0].append([w, h])  # P4
        else:
            scales[1].append([w, h])  # P5
    
    return [np.array(s) for s in scales]

def kmeans_anchors(whs, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(whs)
    centers = kmeans.cluster_centers_
    return centers

if __name__ == "__main__":
    LABEL_DIR = "../training/label_2"  # KITTI 라벨 경로
    IMG_WIDTH = 1245
    IMG_HEIGHT = 384
    
    # P4, P5 그리드 크기 (예: YOLOv4 Tiny KITTI용)
    # 일반적으로 P5는 더 작은 feature map (더 큰 stride)
    # 1280 x 384 이미지 기준
    P4_GRID = (IMG_WIDTH // 16, IMG_HEIGHT // 16)  # 80 x 24
    P5_GRID = (IMG_WIDTH // 32, IMG_HEIGHT // 32)  # 40 x 12
    grid_sizes = [P4_GRID, P5_GRID]

    print("Loading bounding boxes...")
    whs = load_kitti_bboxes(LABEL_DIR)
    print(f"Total boxes loaded: {len(whs)}")

    print("Splitting boxes by scale...")
    scales = split_scales(whs, IMG_WIDTH, IMG_HEIGHT, grid_sizes)
    for i, s in enumerate(scales):
        print(f"Scale {i+4} boxes count: {len(s)}")

    print("Computing k-means anchors...")
    for i, scale_whs in enumerate(scales):
        if len(scale_whs) < 3:
            print(f"Scale {i+4} has too few boxes for clustering")
            continue
        anchors = kmeans_anchors(scale_whs, 3)
        anchors = anchors.astype(int)
        print(f"Anchors for scale {i+4}: {anchors.tolist()}")
