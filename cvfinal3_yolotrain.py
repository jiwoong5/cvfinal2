import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

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
            # KITTI 데이터셋에 최적화된 앵커 (가로가 더 긴 형태)
            if self.img_size[1] > self.img_size[0]:  # width > height (KITTI case)
                self.anchors = [
                    [[10, 13], [16, 30], [33, 23]],  # P4 - 작은 객체용, 스케일 수정
                    [[30, 61], [62, 45], [59, 119]]  # P5 - 큰 객체용, 스케일 수정
                ]
            else:
                # 정사각형 이미지용 기본 앵커
                self.anchors = [
                    [[10, 14], [23, 27], [37, 58]],  # P4
                    [[81, 82], [135, 169], [344, 319]]  # P5
                ]
        else:
            self.anchors = anchors
        
        # Backbone - CSPDarknet53-tiny
        self.conv1 = ConvBNLeaky(3, 32, 3, 2, 1)
        self.conv2 = ConvBNLeaky(32, 64, 3, 2, 1)
        self.csp1 = CSPBlock(64, 64, 1)
        
        self.conv3 = ConvBNLeaky(64, 128, 3, 2, 1)
        self.csp2 = CSPBlock(128, 128, 3)
        
        self.conv4 = ConvBNLeaky(128, 256, 3, 2, 1)
        self.csp3 = CSPBlock(256, 256, 3)
        
        self.conv5 = ConvBNLeaky(256, 512, 3, 2, 1)
        self.csp4 = CSPBlock(512, 512, 1)
        
        # Neck - FPN
        self.conv6 = ConvBNLeaky(512, 256, 1)
        self.conv7 = ConvBNLeaky(256, 512, 3, 1, 1)
        
        # Head - Detection layers
        self.conv8 = ConvBNLeaky(512, 256, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # 수정된 부분: concat 후 512 채널이 입력됨
        self.conv9 = ConvBNLeaky(512, 256, 1)  # 256 + 256 = 512 입력
        self.conv10 = ConvBNLeaky(256, 512, 3, 1, 1)
        
        # Output layers
        self.out1 = nn.Conv2d(512, 3 * (5 + num_classes), 1)  # P5
        self.out2 = nn.Conv2d(512, 3 * (5 + num_classes), 1)  # P4 - conv10 출력이 512 채널
    
    def forward(self, x):
        # Backbone
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.csp1(x)
        
        x = self.conv3(x)
        x = self.csp2(x)
        
        x = self.conv4(x)
        route1 = self.csp3(x)  # P4 route
        
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

class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, img_size=416):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors[0])
        self.num_classes = num_classes
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='sum')
        
        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5
        self.lambda_obj = 1.0
        self.lambda_cls = 1.0
    
    def forward(self, predictions, targets):
        device = predictions[0].device
        total_loss = 0.0
        
        '''
        # 디버깅을 위한 정보 출력
        print(f"Batch size: {len(targets)}")
        non_empty_targets = sum(1 for t in targets if t is not None and len(t) > 0)
        print(f"Non-empty targets: {non_empty_targets}")
        '''
        
        for i, pred in enumerate(predictions):
            batch_size, _, grid_h, grid_w = pred.shape
            stride = self.img_size[0] // grid_h  # height 기준 stride
            
            #print(f"Layer {i}: Grid size {grid_h}x{grid_w}, Stride: {stride}")
            
            pred = pred.view(batch_size, self.num_anchors, 5 + self.num_classes, grid_h, grid_w)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()
            
            # 타겟 생성
            target_tensor = self.build_targets(targets, grid_h, grid_w, i, device, stride)
            
            # 손실 계산
            obj_mask = target_tensor[..., 4] == 1
            noobj_mask = target_tensor[..., 4] == 0
            
            #print(f"Layer {i}: Objects found: {obj_mask.sum().item()}")
            
            # 좌표 손실 (객체가 있는 경우만)
            if obj_mask.sum() > 0:
                pred_xy = torch.sigmoid(pred[..., :2])
                pred_wh = pred[..., 2:4]
                
                target_xy = target_tensor[..., :2]
                target_wh = target_tensor[..., 2:4]
                
                xy_loss = self.mse_loss(pred_xy[obj_mask], target_xy[obj_mask])
                wh_loss = self.mse_loss(pred_wh[obj_mask], target_wh[obj_mask])
                coord_loss = xy_loss + wh_loss
                #print(f"Layer {i}: XY Loss: {xy_loss.item():.4f}, WH Loss: {wh_loss.item():.4f}")
            else:
                coord_loss = 0.0
            
            # 객체성 손실
            if obj_mask.sum() > 0:
                obj_loss = self.bce_loss(pred[obj_mask][..., 4], target_tensor[obj_mask][..., 4])
                #print(f"Layer {i}: Obj Loss: {obj_loss.item():.4f}")
            else:
                obj_loss = 0.0
                
            if noobj_mask.sum() > 0:
                noobj_loss = self.bce_loss(pred[noobj_mask][..., 4], target_tensor[noobj_mask][..., 4])
                #print(f"Layer {i}: NoObj Loss: {noobj_loss.item():.4f}")
            else:
                noobj_loss = 0.0
            
            # 클래스 손실
            if obj_mask.sum() > 0 and self.num_classes > 1:
                cls_loss = self.bce_loss(pred[obj_mask][..., 5:], target_tensor[obj_mask][..., 5:])
                #print(f"Layer {i}: Class Loss: {cls_loss.item():.4f}")
            else:
                cls_loss = 0.0
            
            # 총 손실
            layer_loss = (self.lambda_coord * coord_loss + 
                         self.lambda_obj * obj_loss + 
                         self.lambda_noobj * noobj_loss + 
                         self.lambda_cls * cls_loss)
            
            #print(f"Layer {i}: Total Loss: {layer_loss:.4f}")
            total_loss += layer_loss
        
        final_loss = total_loss / batch_size
        #print(f"Final batch loss: {final_loss:.4f}")
        #print("-" * 50)
        
        return final_loss
    
    def build_targets(self, targets, grid_h, grid_w, layer_idx, device, stride):
        batch_size = len(targets)
        target_tensor = torch.zeros(batch_size, self.num_anchors, grid_h, grid_w, 5 + self.num_classes).to(device)
        
        for b, target in enumerate(targets):
            if target is None or len(target) == 0:
                continue
                
            for obj in target:
                if len(obj) != 5:
                    continue
                    
                cls_id, x_center, y_center, box_width, box_height = obj
                
                # 절대 좌표로 변환 (정규화된 좌표 → 픽셀 좌표)
                abs_x = x_center * self.img_size[1]  # width
                abs_y = y_center * self.img_size[0]  # height
                abs_w = box_width * self.img_size[1]
                abs_h = box_height * self.img_size[0]
                
                # 그리드 좌표로 변환
                gx = abs_x / (self.img_size[1] / grid_w)  # grid x coordinate
                gy = abs_y / (self.img_size[0] / grid_h)  # grid y coordinate
                gw = abs_w / (self.img_size[1] / grid_w)  # grid width
                gh = abs_h / (self.img_size[0] / grid_h)  # grid height
                
                gi = int(gx)
                gj = int(gy)
                
                # 그리드 경계 확인
                if gi >= grid_w or gj >= grid_h or gi < 0 or gj < 0:
                    continue
                
                # 앵커와 가장 잘 매칭되는 것 찾기
                anchor_ious = []
                for anchor in self.anchors[layer_idx]:
                    # 앵커 크기를 현재 그리드 스케일로 변환
                    anchor_w = anchor[0] / (self.img_size[1] / grid_w)
                    anchor_h = anchor[1] / (self.img_size[0] / grid_h)
                    anchor_iou = self.calculate_iou_wh((gw, gh), (anchor_w, anchor_h))
                    anchor_ious.append(anchor_iou)
                
                best_anchor = np.argmax(anchor_ious)
                
                # 타겟 설정
                target_tensor[b, best_anchor, gj, gi, 0] = gx - gi  # x offset
                target_tensor[b, best_anchor, gj, gi, 1] = gy - gj  # y offset
                
                # w, h는 로그 공간에서 계산
                anchor_w = self.anchors[layer_idx][best_anchor][0] / (self.img_size[1] / grid_w)
                anchor_h = self.anchors[layer_idx][best_anchor][1] / (self.img_size[0] / grid_h)
                
                target_tensor[b, best_anchor, gj, gi, 2] = np.log(gw / anchor_w + 1e-16)
                target_tensor[b, best_anchor, gj, gi, 3] = np.log(gh / anchor_h + 1e-16)
                target_tensor[b, best_anchor, gj, gi, 4] = 1  # objectness
                
                # 클래스 원-핫 인코딩
                if self.num_classes > 1:
                    target_tensor[b, best_anchor, gj, gi, 5 + int(cls_id)] = 1
        
        return target_tensor
    
    def calculate_iou_wh(self, wh1, wh2):
        w1, h1 = wh1
        w2, h2 = wh2
        inter_area = min(w1, w2) * min(h1, h2)
        union_area = w1 * h1 + w2 * h2 - inter_area
        return inter_area / (union_area + 1e-16)

class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=(416, 416), transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.transform = transform
        
        # KITTI 클래스 매핑
        self.kitti_classes = {
            'Car': 0, 'Van': 1, 'Truck': 2, 'Pedestrian': 3, 
            'Person_sitting': 4, 'Cyclist': 5, 'Tram': 6, 'Misc': 7
        }
        
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        print(f"Found {len(self.img_files)} images")
        
        # 라벨 파일 존재 확인
        label_count = 0
        for img_file in self.img_files[:10]:  # 처음 10개만 확인
            label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
            label_path = os.path.join(label_dir, label_file)
            if os.path.exists(label_path):
                label_count += 1
                # 라벨 내용 확인
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    print(f"Label {label_file}: {len(lines)} objects")
                    if lines:
                        print(f"First line: {lines[0].strip()}")
        
        print(f"Label files found for first 10 images: {label_count}/10")
    
    def __len__(self):
        return len(self.img_files)
    
    def parse_kitti_label(self, label_path, img_width, img_height):
        """KITTI 라벨을 YOLO 형태로 변환"""
        boxes = []
        
        if not os.path.exists(label_path):
            return boxes
            
        with open(label_path, 'r') as f:
            for line in f.readlines():
                data = line.strip().split()
                if len(data) < 15:  # KITTI 형태는 최소 15개 필드
                    continue
                
                # KITTI 형태: Class truncated occluded alpha bbox_left bbox_top bbox_right bbox_bottom ...
                class_name = data[0]
                
                # 지원하는 클래스만 처리
                if class_name not in self.kitti_classes:
                    continue
                
                # truncated, occluded 값으로 필터링 (선택사항)
                truncated = float(data[1])
                occluded = int(data[2])
                
                # 너무 가려진 객체는 제외 (선택사항)
                if truncated > 0.5 or occluded > 2:
                    continue
                
                # 바운딩 박스 좌표 (픽셀 단위)
                bbox_left = float(data[4])
                bbox_top = float(data[5])
                bbox_right = float(data[6])
                bbox_bottom = float(data[7])
                
                # 유효성 검사
                if bbox_right <= bbox_left or bbox_bottom <= bbox_top:
                    continue
                if bbox_left < 0 or bbox_top < 0 or bbox_right > img_width or bbox_bottom > img_height:
                    continue
                
                # YOLO 형태로 변환 (정규화된 중심 좌표와 크기)
                x_center = (bbox_left + bbox_right) / 2.0 / img_width
                y_center = (bbox_top + bbox_bottom) / 2.0 / img_height
                width = (bbox_right - bbox_left) / img_width
                height = (bbox_bottom - bbox_top) / img_height
                
                # 범위 확인
                if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1:
                    class_id = self.kitti_classes[class_name]
                    boxes.append([class_id, x_center, y_center, width, height])
        
        return boxes
    
    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        
        # 이미지 로드
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image {img_path}")
            return None, None
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_h, original_w = img.shape[:2]
        
        # 라벨 로드 - KITTI 형태를 YOLO 형태로 변환
        label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
        label_path = os.path.join(self.label_dir, label_file)
        
        boxes = self.parse_kitti_label(label_path, original_w, original_h)
        
        # 이미지 리사이즈 - KITTI는 종횡비 유지 중요
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]))  # (width, height)
        img = img.astype(np.float32) / 255.0
        
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1)
        
        return img, boxes

def collate_fn(batch):
    """배치 처리를 위한 커스텀 collate 함수"""
    batch = [b for b in batch if b[0] is not None]  # None 제거
    if len(batch) == 0:
        return None, None
    
    images, targets = zip(*batch)
    images = torch.stack(images)
    return images, list(targets)

def train_yolov4_tiny(model, train_loader, val_loader=None, num_epochs=100, device='cuda'):
    """YOLOv4 Tiny 모델 학습 함수"""
    
    # 옵티마이저 및 스케줄러 설정
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)  # 학습률 낮춤
    
    # val_loader가 없으면 train loss 기반 스케줄러 사용
    if val_loader is not None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # 손실 함수
    criterion = YOLOLoss(model.anchors, model.num_classes, model.img_size)
    
    # 학습 기록
    train_losses = []
    val_losses = [] if val_loader is not None else None
    
    model.to(device)
    
    for epoch in range(num_epochs):
        # 학습 모드
        model.train()
        epoch_train_loss = 0
        valid_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training')
        for batch_idx, (images, targets) in enumerate(train_pbar):
            if images is None:  # 배치가 비어있는 경우
                continue
                
            images = images.to(device)
            
            # Forward pass
            predictions = model(images)
            loss = criterion(predictions, targets)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss at batch {batch_idx}")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_train_loss += loss.item()
            valid_batches += 1
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            # 처음 몇 에포크는 더 자주 출력
            if epoch < 3 and batch_idx < 5:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}: Loss = {loss.item():.4f}")
        
        if valid_batches == 0:
            print(f"Warning: No valid batches in epoch {epoch+1}")
            continue
            
        # 평균 학습 손실 계산
        avg_train_loss = epoch_train_loss / valid_batches
        train_losses.append(avg_train_loss)
        
        # 검증 (val_loader가 있는 경우만)
        avg_val_loss = None
        if val_loader is not None:
            model.eval()
            epoch_val_loss = 0
            valid_val_batches = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation')
                for images, targets in val_pbar:
                    if images is None:
                        continue
                        
                    images = images.to(device)
                    predictions = model(images)
                    loss = criterion(predictions, targets)
                    
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        epoch_val_loss += loss.item()
                        valid_val_batches += 1
                        val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            if valid_val_batches > 0:
                avg_val_loss = epoch_val_loss / valid_val_batches
                val_losses.append(avg_val_loss)
                
                # 스케줄러 업데이트 (validation loss 기반)
                scheduler.step(avg_val_loss)
                
                print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
                
                # 모델 저장 (최고 성능일 때)
                if epoch == 0 or avg_val_loss < min(val_losses[:-1]):
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': avg_train_loss,
                        'val_loss': avg_val_loss,
                    }, f'yolov4_tiny_best.pth')
                    print(f'Model saved at epoch {epoch+1}')
        
        else:
            # validation 없는 경우
            scheduler.step()  # StepLR 업데이트
            
            print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}')
            
            # 주기적 모델 저장 (validation 없으므로)
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                }, f'yolov4_tiny_epoch_{epoch+1}.pth')
                print(f'Model saved at epoch {epoch+1}')
    
    return train_losses, val_losses

def plot_training_curves(train_losses, val_losses=None):
    """학습 곡선 시각화"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    
    if val_losses is not None:
        plt.plot(val_losses, label='Validation Loss', color='red')
    
    plt.title('YOLOv4 Tiny Training Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# 사용 예시
if __name__ == "__main__":
    # 하이퍼파라미터 설정 - KITTI 데이터셋용
    IMG_SIZE = (384, 1280)  # (height, width) - KITTI 원본 비율 유지
    BATCH_SIZE = 4  # 배치 크기 더 감소
    NUM_CLASSES = 8  # KITTI 클래스: Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc
    NUM_EPOCHS = 30
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 데이터셋 경로 - Validation 없이 학습 가능
    TRAIN_IMG_DIR = "../data_object_image_2/training/image_2"
    TRAIN_LABEL_DIR = "../training/label_2"
    
    # Validation이 있는 경우 (선택사항)
    VAL_IMG_DIR = None  # None으로 설정하면 validation 건너뜀
    VAL_LABEL_DIR = None
    
    # 데이터 변환
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # 데이터셋 및 데이터로더 생성
    train_dataset = YOLODataset(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, IMG_SIZE, transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=2, collate_fn=collate_fn)  # num_workers 감소
    
    # Validation 데이터로더 (선택사항)
    val_loader = None
    if VAL_IMG_DIR and VAL_LABEL_DIR and os.path.exists(VAL_IMG_DIR):
        val_dataset = YOLODataset(VAL_IMG_DIR, VAL_LABEL_DIR, IMG_SIZE, transform)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                               num_workers=2, collate_fn=collate_fn)
        print(f"Validation batches: {len(val_loader)}")
    else:
        print("Training without validation set")
    
    print(f"Training batches: {len(train_loader)}")
    
    # 모델 생성
    model = YOLOv4Tiny(num_classes=NUM_CLASSES, img_size=IMG_SIZE)
    
    print(f"Training on device: {DEVICE}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 학습 실행
    train_losses, val_losses = train_yolov4_tiny(
        model, train_loader, val_loader, NUM_EPOCHS, DEVICE
    )
    
    # 학습 곡선 시각화
    plot_training_curves(train_losses, val_losses)
    
    print("Training completed!")