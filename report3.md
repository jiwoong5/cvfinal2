### 1. 서론

객체 인식이란?
- 이미지나 영상 내에서 특정 객체를 자동으로 찾아내는 기술
- 사진 속에 사람, 자동차, 개, 고양이 등의 객체를 구분하여 인식하는 것
- **객체 분류**: 이미지를 통해 객체의 종류를 식별
- **객체 로컬라이제이션**: 이미지 내에서 객체의 위치를 사각형(바운딩 박스)으로 지정

YOLO(You Only Look Once)란?
- 객체 인식의 하나의 모델로, 이미지 내에서 객체를 동시에 여러 개 인식할 수 있도록 설계된 알고리즘
- 실시간 객체 인식을 목표로 하는 단일 단계 모델
- **단일 네트워크**: 기존의 객체 인식 방법은 보통 후보 영역을 여러 번 검사하는 방식(예: R-CNN)을 사용하지만, YOLO는 이미지를 한 번에 보고 객체를 예측
- **실시간 처리**: YOLO는 매우 빠르기 때문에 실시간 객체 인식에 적합합니다.
- **고속**: 이미지 전체를 한 번에 처리하기 때문에 다른 객체 인식 알고리즘보다 속도가 빠릅니다.
- 전체 맥락을 고려: 이미지를 한 번에 처리하면서도 전역적인 맥락을 잘 반영합니다.

### 2. 실험 개요
본 실험에서는 yolo v4 tiny 모델을 참고해 yolo model을 설계하고 kitti 데이터셋을 이용해 이를 학습시킴

YOLOv4 Tiny
- 경량화된 YOLOv4 모델로 빠른 추론 속도를 위해 네트워크 깊이와 너비를 줄임
- 정확도는 YOLOv4보다 낮지만, 속도가 매우 빠름
- 입력 크기: 보통 416x416 또는 320x320 (더 작은 입력으로 속도 향상 가능)
- Feature Extraction: 깊이와 채널 수가 줄어든 CSPDarknet53 Lite
- Skip Connections: YOLOv4처럼 완전하지 않고, 제한적으로만 적용
- Detection Head: 2개 scale에서 예측 (주로 13x13, 26x26 feature map)
- Anchor Boxes: 미리 정의된 anchor box 크기를 사용해 예측
  
KITTI 데이터셋
- 자율주행용 객체 탐지 벤치마크 데이터셋
- label format: .txt 형식 ex)Pedestrian 0.00 0 -3.05 499.95 155.21 566.95 309.56 1.75 0.67 1.13 -0.88 1.47 8.43 3.13
- Class: 8~10개. car, van, truck, pedestrian, person_sitting, cyclist, tram, misc
- 이미지 크기: 375 x 1245

설계 목표
- 경량화: YOLOv4 대비 적은 파라미터와 연산량으로 모델 경량화 달성
- 속도 최적화: 실시간 객체 탐지를 위한 빠른 추론 속도 확보 (초당 프레임 수 향상)
- 기본 구조 유지: YOLOv4의 핵심 아이디어(Anchor 기반 검출, CSPDarknet Backbone 등)를 간소화하여 유지
- 효율적 Feature Extraction: 경량화된 CSPDarknet53 Lite Backbone 사용 및 2단계 스케일 Detection
Head 구성

### 3. 실험 조건
이미지 정규화
- 기존 KITTI 입력 크기 이해: KITTI 원본 이미지는 보통 1245x375 (가로x세로) 정도로 비율이 직사각형.
- 크기유지장점: 해상도 보존, 정확도 잠재력, 비율 왜곡 없음
- 크기유지단점: 모델 구조 제한, 계산량 증가, 모델 아키텍처 수정 필요
- 네트워크, 특히 YOLO 계열 모델은 보통 정사각형 (예: 416x416, 512x512) 입력 크기를 선호
- 본 실험에서는 해상도 보존과 객체 탐지 정확도에 초점을 두어 정규화하지 않은 이미지크기를 입력으로 사용

label 활용
- 기존 KITTI label 구조: 클래스이름, truncated, occluded, alpha, 2d 바운딩 박스, 3d 크기, 3d 위치, rotation_y
- 이 중 클래스 이름, truncated, occluded, 2d 바운딩 박스를 사용

anchor
```
self.anchors = [
                    [[40, 32], [228, 122], [107, 83]],  # P4
                    [[395, 187], [300, 193], [354, 348]]  # P5
                ]
```
- tiny모델과 같이 p3, p4, p5 3개의 스케일을 사용하지 않고 p4, p5 2개의 스케일만을 사용
- kmeans 방식을 사용해 최적 anchor 추출

coord_loss (좌표 손실)
- 예측된 박스의 중심 좌표(x, y)와 크기(w, h)가 실제 객체와 얼마나 일치하는지 측정
- 중심 좌표(x, y)는 MSELoss(평균 제곱 오차)로 측정 (예측값과 실제값 차이 제곱합)
- 너비와 높이(w, h)는 로그 공간에서 MSE 계산 (크기 차이를 비율로 더 잘 반영)
- 적용 대상: 실제 객체가 있는 앵커 위치에 대해서만 계산

obj_loss (객체성 손실)
- 해당 위치에 실제 객체가 있을 때, 모델이 객체 존재 확률을 얼마나 정확히 예측했는지 평가
- 계산 방식: BCEWithLogitsLoss (바이너리 크로스 엔트로피 손실)
- 적용 대상: 객체가 있는 위치 (objectness score에 대해)
  
noobj_loss (비객체성 손실)
- 객체가 없는 위치에서 모델이 객체 존재하지 않음을 잘 예측하는지 평가
- 계산 방식: BCEWithLogitsLoss
- 적용 대상: 객체가 없는 위치

cls_loss (클래스 손실)
- 예측된 객체 클래스가 실제 클래스와 얼마나 맞는지 평가
- 계산 방식: BCEWithLogitsLoss를 다중 클래스 원-핫 벡터에 대해 계산 (다중 클래스 분류)
- 적용 대상: 객체가 있는 위치

총 손실 계산
```
layer_loss = lambda_coord * coord_loss
           + lambda_obj * obj_loss
           + lambda_noobj * noobj_loss
           + lambda_cls * cls_loss
```

### 4. 주요 모델 구조 설명
YOLOv4 Tiny 입력
- RGB 이미지 입력 (3채널)
- 이미지 크기: 기본 416x416 (튜플 형태 가능)

YOLOv4 Tiny Backbone (특징 추출)
- ConvBNLeaky(3→32), stride=2 → 다운샘플링 시작
- ConvBNLeaky(32→64), stride=2 → 더 작은 공간 표현
- CSPBlock(64→64, num_blocks=1)
- ConvBNLeaky(64→128), stride=2 → 다운샘플링
- CSPBlock(128→128, num_blocks=3)
- ConvBNLeaky(128→256), stride=2 → 다운샘플링
- CSPBlock(256→256, num_blocks=3) → 이 출력이 중간 특징 맵 (route1, P4 스케일)
- ConvBNLeaky(256→512), stride=2 → 다운샘플링
- CSPBlock(512→512, num_blocks=1)
- 포인트:CSPBlock 은 채널 분할 후 병렬 합치기 구조로, 계산 효율과 표현력 향상

YOLOv4 Tiny Neck (특징 융합, FPN 역할)
- ConvBNLeaky(512→256), kernel=1 (채널 축소)
- ConvBNLeaky(256→512), kernel=3, padding=1 (특징 재처리)

YOLOv4 Tiny Head (검출 레이어) 첫 번째 출력 (P5 scale)
- ConvBNLeaky(512→256), kernel=1
- out1: Conv2d(512 → 3 * (5 + num_classes), kernel=1)
- 가장 낮은 해상도(작은 피처맵)에서 큰 객체 검출 담당

YOLOv4 Tiny Head (검출 레이어) 두 번째 출력 (P4 scale)
- ConvBNLeaky(256→256), kernel=1
- 업샘플링(scale_factor=2) → 공간 크기 2배 확대
- Concatenate 업샘플링된 특징 + route1 (256채널) → 512채널 입력
- ConvBNLeaky(512→256), kernel=1
- ConvBNLeaky(256→512), kernel=3, padding=1
- out2: Conv2d(512 → 3 * (5 + num_classes), kernel=1)
- 중간 해상도 피처맵에서 작은 객체 검출 담당

YOLOv4 Tiny 출력 형태
- out1, out2: 각각 (batch, 3 * (5 + num_classes), grid_h, grid_w) 형태
- 3: 앵커 개수
- 5: (x, y, w, h, objectness)

### 5. 실험 결과
10개의 random sampling 된 이미지에 대해 객체 인식 수행

<img src="https://github.com/jiwoong5/cvfinal2/blob/main/3_box_analysis.png" width="1200" height="600" />

- width 분포: gt 박스와 추론된 박스 width 빈도
- heighth 분포: gt 박스와 추론된 박스 height 빈도
- area 분포: gt 박스와 추론된 박스 area 빈도
- width + height 분포: gt 박스와 추론된 박스 height + width 빈도
- average width by class: gt 박스와 추론된 박스 클래스 별 평균 width
- 사용중인 anchor 와 샘플링된 이미지 width + height 비교
  
<img src="https://github.com/jiwoong5/cvfinal2/blob/main/3_comparison.png" width="1200" height="600" />

- 생성된 박스를 시각화하여 gt와 비교

<img src="https://github.com/jiwoong5/cvfinal2/blob/main/3_quantity.png" width="1200" height="600" />

- 정량적 요소로 모델 평가
- 
### 6. 분석 및 고찰

- 본 연구에서는 YOLOv4 Tiny 모델을 이용하여 KITTI 데이터셋 상의 객체 검출 성능을 정성적 및 정량적으로 평가하였다. 실험 결과 다음과 같은 주요 특징을 확인할 수 있었음
- GT 바운딩 박스와의 유사성: 예측된 바운딩 박스들은 전반적으로 Ground Truth(GT) 객체의 폭(width) 및 높이(height) 크기를 잘 추종하는 경향을 보임 이는 모델이 객체의 크기 정보를 일정 수준 이상으로 학습했음을 의미
- 근거리 객체에 대한 오탐지: 카메라에 가까이 위치한 객체(예: 프레임 전면부의 차량)의 경우, 실제 바운딩 박스의 크기가 크고, 이미지 내에서도 Outlier로 분류될 만큼 비정상적인 크기를 가지게 됨
- 이러한 객체들에 대해 모델의 탐지 성능이 저하되는 현상이 나타났으며, 이는 모델이 일반적인 크기의 객체에 더 민감하게 반응하도록 학습되었기 때문으로 추정
- 배경과의 구분 어려움: 일부 객체는 복잡한 배경(예: 차량 정체 구간, 나무 배경 등)과 겹쳐지면서 오탐지 또는 미탐지 사례가 발생하였다. 이는 모델의 특징 추출 능력 한계 혹은 NMS 처리 단계에서의 누락 가능성을 시사.
- 이러한 결과는 모델이 일반적인 객체 크기에서는 일정 수준 이상의 성능을 보이지만, 극단적인 크기나 배경 복잡도가 높은 경우에는 추가적인 보완이 필요함을 보여줌
  
7. 결론 및 향후 연구
결론
- 본 연구에서는 YOLOv4 Tiny 모델을 기반으로 KITTI 데이터셋에 대한 객체 검출 성능을 정성적 및 정량적으로 평가
- 모델은 경량화된 구조를 바탕으로 비교적 적은 수의 파라미터를 가지며, 이에 따라 적은 수의 에포크로도 빠르게 수렴하는 경향을 보임.
- 실험 결과, 30 에포크 내에서도 일정 수준 이상의 정확도와 재현율을 확보할 수 있었으며, 이는 실시간 응용을 고려한 경량 모델로서의 가능성을 보여줌
- anchor box의 크기를 변경하여 실험한 결과, 전체 성능에 미치는 영향은 제한적이었으며, 이는 YOLOv4 Tiny 구조가 anchor 설정에 대해 상대적으로 덜 민감하게 반응함을 시사

향후 연구 방향
- 고정된 입력 해상도 제한 극복: 현재 모델은 고정 해상도 입력을 기반으로 학습 및 추론이 이루어졌으나, 다양한 해상도에 대한 유연성을 높이기 위한 다중 해상도 학습 기법 적용이 필요
- 데이터셋 확장 및 일반화 성능 분석: KITTI 외의 다양한 도메인(예: 도시 주행, 야간 환경 등)에서의 성능을 평가하고, 데이터 일반화 능력을 분석할 필요가 있음
- Anchor-free 방식 비교 연구: anchor 기반 YOLO 계열 모델과 최근 주목받는 anchor-free 기반 모델 간의 성능 및 효율성 비교를 통해 최적의 경량 객체 검출 방식을 탐색할 수 있음.
- 후처리 최적화: 현재의 NMS 기반 후처리를 보다 고도화하거나, 학습 기반의 Soft-NMS 또는 DIoU-NMS와 같은 대안을 적용하여 성능 개선 가능성을 모색할 수 있음
