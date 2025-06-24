# Mono Depth Estimation 모델 비교 실험 보고서

## 1. 서론
UNetDepth란?
- UNetDepth는 UNet 구조를 기반으로 한 단안 깊이 추정 모델로, 입력 이미지에서 픽셀 단위 깊이 맵을 예측
- UNet은 인코더-디코더 구조를 갖추고 있으며, 인코더는 특징 추출, 디코더는 해상도 복원을 담당
- 깊이 추정에 적합하도록 설계된 UNetDepth는 공간 정보를 효과적으로 활용하여 정확한 깊이 맵 생성이 가능

기본 UNetDepth 구조의 문제점
- 단순 인코더-디코더 연결만으로는 저수준 공간 정보가 깊이 맵 복원에 충분히 반영되지 않을 수 있음
- 네트워크가 깊어질수록 역전파 시 기울기 소실(Gradient Vanishing) 문제로 학습이 어려워질 수 있음
- 배치 간 **통계 변동**으로 인한 학습 불안정성 문제가 존재할 수 있음

Batch 간 통계 변동이란?
- 배치 간 통계 변동은 서로 다른 미니배치에서 계산되는 통계값(평균, 분산)이 서로 달라지는 현상
- 어떤 배치에서는 특정 특징이 평균보다 높게 나타나지만, 다른 배치에서는 낮게 나타날 수 있음
- 배치마다 통계가 변동하면, 모델이 정규화된 출력의 분포가 매번 달라지는 환경에서 학습
- 배치 크기가 너무 작거나, 데이터 분포가 크게 다를 때 이 현상이 두드러짐
- 실험조건의 배치 크기 8은 작은 편에 속함.
  
## 2. 실험 개요
본 실험에서는 단안 깊이 추정(Monocular Depth Estimation) 모델의 세 가지 변형을 비교
- **Original**: 기본 UNetDepth 구조  
- **NoSkip**: Skip Connection을 제거한 모델  
- **BatchNorm**: 배치 정규화(Batch Normalization)를 추가한 모델  

단일 깊이 추정이란?
- **단일 깊이 추정(Monocular Depth Estimation)**: 단일 이미지(즉, 한 대의 카메라로 찍은 사진)에서 각 픽셀의 깊이 정보를 예측하는 기술
- 같은 2D 이미지에서 다양한 3D 장면이 만들어질 수 있어, 단일 이미지에서 깊이를 정확히 추정하는 건 본질적으로 어려운 문제
- 기하학적 단서 이용: 명암, 질감, 물체의 크기, 투영 왜곡 등의 2D 이미지 단서를 이용해 깊이를 추론

CNN을 활용한 단일 깊이 추정
- 대규모 이미지-깊이 쌍 데이터셋(예: KITTI, NYU Depth)을 이용해 딥러닝 모델을 학습
- 모델은 입력 이미지에서 특징을 추출하고, 각 픽셀마다 깊이를 예측

Connection (연결)이란?
- 신경망에서 connection은 한 층(layer)의 출력을 다음 층으로 전달하는 통로를 의미
- 일반적으로 신경망은 층들이 순차적으로 연결되어 있고, 한 층의 출력이 바로 다음 층의 입력
- **Skip Connection**: 어떤 층의 출력을 다음 층뿐 아니라, 그보다 더 뒤쪽 층으로 바로 연결하는 것
- 중간 층 몇 개를 건너뛰고 출력을 전달하는 연결
- CNN을 통한 학습에서 Gradient Vanishing 문제의 해결책으로 사용될 수 있음

Batch Normalization 이란?
- 학습 중 미니배치 단위로 입력 데이터(특징 맵)의 평균과 분산을 계산하여 정규화
- 한 배치(batch)에 속한 샘플들의 통계(평균, 분산)를 이용해 그 배치 내 데이터를 정규화
- 학습 중 배치별 통계로 정규화를 하고, 추론 시에는 학습 전체에서 축적한 평균과 분산의 이동평균(running mean & variance) 을 사용하여 고정된 분포로 정규화를 수행

## 3. 실험 조건

실험 조건
- **배치 크기 (batch_size)**: 8
- **워커 수 (num_workers)**: 2
- **에폭 수 (num_epochs)**: 20 (빠른 학습을 위해 축소)

이미지 전처리 (transform_img)
- 이미지 크기: 224 x 224로 리사이즈
- 텐서 변환
- 정규화 (평균: [0.485, 0.456, 0.406], 표준편차: [0.229, 0.224, 0.225])

feature
- [64,128,256,512]

  
컨볼루션 블록
```
def _conv_block(self, in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True),
    )
```
- 두 번의 Conv2D + ReLU로 비선형 표현력을 높이고, 특징 추출 강화
- padding=1 덕분에 출력의 spatial 크기(height, width)는 유지됨
- 연속된 두 Conv 레이어를 사용하면 단일 Conv보다 더 복잡한 패턴 인식 가능
- pixel 기준 3 by 3 범위를 보고 feature 추출

UNET forward 구조
```
def forward(self, x):
      e1 = self.enc1(x)               # [B, 64, H, W]
      e2 = self.enc2(self.pool1(e1))  # [B, 128, H/2, W/2]
      e3 = self.enc3(self.pool2(e2))  # [B, 256, H/4, W/4]
      e4 = self.enc4(self.pool3(e3))  # [B, 512, H/8, W/8]

      b  = self.bottleneck(self.pool4(e4))  # [B, 1024, H/16, W/16]

      d4 = self.up4(b)                      # [B, 512, H/8, W/8]
      d4 = torch.cat((d4, e4), dim=1)       # 채널 합쳐서 [B, 1024, H/8, W/8]
      d4 = self.dec4(d4)                    # 다시 [B, 512, H/8, W/8]

      d3 = self.up3(d4)                     # [B, 256, H/4, W/4]
      d3 = torch.cat((d3, e3), dim=1)       # [B, 512, H/4, W/4]
      d3 = self.dec3(d3)                    # [B, 256, H/4, W/4]

      d2 = self.up2(d3)                     # [B, 128, H/2, W/2]
      d2 = torch.cat((d2, e2), dim=1)       # [B, 256, H/2, W/2]
      d2 = self.dec2(d2)                    # [B, 128, H/2, W/2]

      d1 = self.up1(d2)                     # [B, 64, H, W]
      d1 = torch.cat((d1, e1), dim=1)       # [B, 128, H, W]
      d1 = self.dec1(d1)                    # [B, 64, H, W]

      return self.act(self.conv_last(d1))
```
- enc1: 64 채널 출력
- enc2: 128 채널 출력
- enc3: 256 채널 출력
- enc4: 512 채널 출력
- bottlenect: 512 채널 출력
- 채널 수 변화: 3 → 64 → 128 → 256 → 512 → 1024 → 512 → 256 → 128 → 64 → 1

## 4. 주요 모델 구조 설명
공통 구조 개요
- UNet 계열 모델은 대개 다음 4가지 단계로 구성: 인코더 (Encoder): 이미지의 점진적 압축 → 더 넓은 수용영역으로 고차원 특징 추출
- 보틀넥 (Bottleneck): 가장 추상적인 feature 추출
- 디코더 (Decoder): 해상도 복원 (업샘플링)
- 마지막 출력 (Output Conv): 깊이 맵 등 목적에 맞는 단일 채널 생성

No Skip 디코더
```
self.up4 = nn.ConvTranspose2d(...)
self.dec4 = self._conv_block(...)
```
- 업샘플 후 바로 디코더 block에 입력
- Skip 연결 없음 → 단일 정보만 활용

BN conv block
```
nn.Conv2d(...)
nn.BatchNorm2d(...)
nn.ReLU()
```
- Convolution 후 BN → 비선형 활성화
- 각 단계마다 출력값 분포 정규화 → 그래디언트 흐름 원활

## 4. 실험 결과

### 4.1 정량적 평가

<img src="https://github.com/jiwoong5/cvfinal2/blob/main/2_%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%85%E1%85%A3%E1%86%BC%E1%84%8C%E1%85%A5%E1%86%A8%E1%84%87%E1%85%B5%E1%84%80%E1%85%AD.png" width="1200" height="600" />

- BatchNorm은 학습을 안정화하고 일반화 성능을 높여, 가장 낮은 MAE와 RMSE를 달성
- Skip Connection은 디코더가 세부적인 공간 정보를 복원하는 데 중요하며, 제거 시 성능이 하락
  
### 4.2 정성적 평가

- 원본 이미지와 Ground Truth 깊이 맵 대비 각 모델의 예측 깊이 맵 시각화  
- Original 모델은 Skip Connection을 통한 정보 전달로 전체적인 깊이 표현이 비교적 정확함  
- NoSkip 모델은 세밀한 깊이 정보 손실이 관찰됨  
- BatchNorm 모델은 학습 안정성과 세밀한 깊이 복원이 우수함

## 5. 분석 및 고찰

- **Skip Connection**: 깊이 추정에서 중요한 공간 정보가 네트워크 깊은 층까지 전달되는 역할을 하므로, 제거 시 정보 손실이 발생해 성능 저하를 유발  
- **Batch Normalization**: 각 층의 입력 분포를 정규화하여 학습 안정성 및 수렴 속도를 개선하며, 깊은 네트워크에서 과적합 감소 효과 기대  
- **에포크 증가**: BatchNorm 모델에서 200 에포크로 장기 학습 시 RMSE가 크게 감소하여 학습 안정성과 표현력 향상을 확인  

## 6. 결론 및 향후 연구

- 모델 구성 변경에 따라 Mono Depth Estimation 성능에 유의미한 차이가 나타났다.  
- Batch Normalization과 충분한 학습 시간 확보가 깊이 추정의 정확도 향상에 효과적임을 확인했다.  
- 향후 다양한 정규화 기법과 구조 변형(예: Attention, Residual Block) 도입을 통한 추가 성능 개선 연구가 필요하다.

