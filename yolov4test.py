import glob
import random
import cv2
import numpy as np
from collections import defaultdict

# ----------------------------
# CONFIG
# ----------------------------
cfg_path = 'yolov4-tiny.cfg'
weights_path = 'yolov4-tiny.weights'
names_path = 'coco.names'
image_dir = '../data_object_image_2/training/image_2'
label_dir = '../training/label_2'

# ----------------------------
# Load class names
# ----------------------------
with open(names_path, 'r') as f:
    classes = f.read().strip().split('\n')

# ----------------------------
# Load YOLOv4-tiny model
# ----------------------------
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# ----------------------------
# Load image samples
# ----------------------------
image_paths = glob.glob(f'{image_dir}/*.png')
sampled_paths = random.sample(image_paths, 10)

# ----------------------------
# Utility: IoU
# ----------------------------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

# ----------------------------
# Utility: Load GT boxes (KITTI format)
# ----------------------------
def load_ground_truth(txt_path, class_names):
    boxes = []
    try:
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                class_name = parts[0]
                if class_name not in class_names:
                    continue
                cls_id = class_names.index(class_name)
                x1 = float(parts[4])
                y1 = float(parts[5])
                x2 = float(parts[6])
                y2 = float(parts[7])
                boxes.append([cls_id, x1, y1, x2, y2])
    except FileNotFoundError:
        pass
    return boxes

# ----------------------------
# Step 1~3: Inference + GT Load
# ----------------------------
yolo_preds = defaultdict(list)  # image_path -> list of [cls, conf, x1, y1, x2, y2]
gt_boxes = defaultdict(list)    # image_path -> list of [cls, x1, y1, x2, y2]

for path in sampled_paths:
    image = cv2.imread(path)
    height, width = image.shape[:2]

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
                cx, cy, w, h = detection[0:4] * np.array([width, height, width, height])
                x1 = int(cx - w / 2)
                y1 = int(cy - h / 2)
                x2 = int(cx + w / 2)
                y2 = int(cy + h / 2)
                boxes.append([x1, y1, x2, y2])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

    # OpenCV 버전 및 빈 결과 대응
    if isinstance(indices, tuple) or len(indices) == 0:
        indices = indices[0] if len(indices) > 0 else []
    elif isinstance(indices, np.ndarray):
        indices = indices.flatten().tolist()
    else:
        indices = list(indices)

    # 예측 박스 시각화
    for i in indices:
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Load GT
    image_id = path.split('/')[-1].replace('.png', '')
    label_path = f"{label_dir}/{image_id}.txt"
    gt_boxes[path] = load_ground_truth(label_path, classes)

# ----------------------------
# Step 4: Evaluate mAP (간단 버전, per-class 평균 정밀도)
# ----------------------------
from collections import Counter

def evaluate_map(yolo_preds, gt_boxes, iou_thresh=0.5):
    tp_fp_counter = defaultdict(lambda: {'tp': 0, 'fp': 0, 'gt': 0})

    for img_path in yolo_preds:
        preds = yolo_preds[img_path]
        gts = gt_boxes[img_path]
        matched = set()

        for pred in sorted(preds, key=lambda x: -x[1]):  # sort by confidence
            pred_cls, conf, px1, py1, px2, py2 = pred
            pred_box = [px1, py1, px2, py2]
            found_match = False

            for i, gt in enumerate(gts):
                gt_cls, gx1, gy1, gx2, gy2 = gt
                if gt_cls == pred_cls and i not in matched:
                    gt_box = [gx1, gy1, gx2, gy2]
                    if iou(pred_box, gt_box) >= iou_thresh:
                        tp_fp_counter[pred_cls]['tp'] += 1
                        matched.add(i)
                        found_match = True
                        break

            if not found_match:
                tp_fp_counter[pred_cls]['fp'] += 1

        # 카운트 GT 수
        gt_class_counts = Counter([gt[0] for gt in gts])
        for cls_id, count in gt_class_counts.items():
            tp_fp_counter[cls_id]['gt'] += count

    aps = []
    for cls_id in tp_fp_counter:
        tp = tp_fp_counter[cls_id]['tp']
        fp = tp_fp_counter[cls_id]['fp']
        gt = tp_fp_counter[cls_id]['gt']
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (gt + 1e-6)
        ap = precision * recall  # 간단한 근사
        print(f"[{classes[cls_id]}] Precision: {precision:.3f}, Recall: {recall:.3f}, AP: {ap:.3f}")
        aps.append(ap)

    mAP = sum(aps) / len(aps) if aps else 0.0
    print(f"\n📊 mean AP (mAP) @ IoU {iou_thresh}: {mAP:.4f}")
    return mAP

# ----------------------------
# Step 5: Run Evaluation
# ----------------------------
evaluate_map(yolo_preds, gt_boxes)
