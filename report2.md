# Mono Depth Estimation 모델 비교 실험 보고서

## 1. 서론
UNetDepth란?
- UNetDepth는 UNet 구조를 기반으로 한 단안 깊이 추정 모델로, 입력 이미지에서 픽셀 단위 깊이 맵을 예측
- UNet은 인코더-디코더 구조를 갖추고 있으며, 인코더는 특징 추출, 디코더는 해상도 복원을 담당
- 깊이 추정에 적합하도록 설계된 UNetDepth는 공간 정보를 효과적으로 활용하여 정확한 깊이 맵 생성이 가능
- skip connection 구조: low-level 정보를 later stage까지 직접 전달해서 보완하는 잔차구조

기본 UNetDepth 구조의 문제점
- 단순 인코더-디코더 연결만으로는 저수준 공간 정보가 깊이 맵 복원에 충분히 반영되지 않을 수 있음
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
  
UNet Base
```
class UNetDepthBase(nn.Module):
    def __init__(self, in_channels=3, features=[64,128,256,512], use_skip=True, use_bn=False):
        super().__init__()
        self.use_skip = use_skip
        self.features = features
        self.enc1 = self._conv_block(in_channels, features[0], use_bn)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._conv_block(features[0], features[1], use_bn)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._conv_block(features[1], features[2], use_bn)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = self._conv_block(features[2], features[3], use_bn)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = self._make_bottleneck(features[3], use_bn)

        self.up4 = nn.ConvTranspose2d(features[3]*2, features[3], 2, 2)
        self.dec4 = self._conv_block(features[3]*2 if use_skip else features[3], features[3], use_bn)

        self.up3 = nn.ConvTranspose2d(features[3], features[2], 2, 2)
        self.dec3 = self._conv_block(features[2]*2 if use_skip else features[2], features[2], use_bn)

        self.up2 = nn.ConvTranspose2d(features[2], features[1], 2, 2)
        self.dec2 = self._conv_block(features[1]*2 if use_skip else features[1], features[1], use_bn)

        self.up1 = nn.ConvTranspose2d(features[1], features[0], 2, 2)
        self.dec1 = self._conv_block(features[0]*2 if use_skip else features[0], features[0], use_bn)

        self.conv_last = nn.Conv2d(features[0], 1, 1)
        self.act = nn.ReLU()

    def _conv_block(self, in_ch, out_ch, use_bn):
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        ]
        return nn.Sequential(*layers)

    def _make_bottleneck(self, in_ch, use_bn):
        return self._conv_block(in_ch, in_ch * 2, use_bn)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat((d4, e4), dim=1) if self.use_skip else d4)

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat((d3, e3), dim=1) if self.use_skip else d3)

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat((d2, e2), dim=1) if self.use_skip else d2)

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat((d1, e1), dim=1) if self.use_skip else d1)

        return self.act(self.conv_last(d1))
```
인코더 (Encoder blocks)
- self.enc1 ~ self.enc4 : 4단계 인코더 블록, 각각 채널 수가 증가 (예: 64 → 128 → 256 → 512)
- self.pool1 ~ self.pool4 : 각 인코더 단계 뒤에 있는 MaxPool2d(2)로 공간 크기를 절반씩 줄임
- 각 인코더 블록은 _conv_block 메서드를 사용해 구성

**conv_block 메서드**
- 2개의 3x3 컨볼루션 + 활성화 레이어로 구성
- use_bn=True면 각 컨볼루션 뒤에 배치 정규화 포함

병목 부분 (Bottleneck)
- self.bottleneck : 가장 깊은 층에서 인코더의 마지막 출력 채널(예: 512)을 두 배(1024)로 늘린 컨볼루션 블록

디코더 부분 (Decoder blocks)
- self.up4, self.up3, self.up2, self.up1 : ConvTranspose2d를 이용한 업샘플링 (업컨볼루션)
- 예: 채널 1024 → 512, 512 → 256 등
- 공간 크기는 두 배로 증가
- self.dec4, self.dec3, self.dec2, self.dec1 : 업샘플링 후 처리할 컨볼루션 블록
- 스킵 연결 사용 시 : 인코더의 동일 단계 출력과 채널을 합쳐서(Concatenate) 채널 수가 두 배가 됨
- 스킵 연결 미사용 시 : 인코더 피쳐 없이 업샘플링된 결과만 사용 (채널 수는 원래 업샘플링된 채널 수)

최종 출력 레이어
- self.conv_last : 1x1 컨볼루션, 디코더 마지막 출력 채널을 1 (예: 깊이 맵)로 축소
- self.act : ReLU 활성화 함수 (출력값을 0 이상으로 만듦)

## 4. 주요 모델 구조 설명
Original UNet
```
class UNetDepth(UNetDepthBase):
    def __init__(self, in_channels=3, features=[64,128,256,512]):
        super().__init__(in_channels, features, use_skip=True, use_bn=False)
```

- use_skip=True: 스킵 연결(skip connection)을 사용
- use_bn=False: 배치 정규화(Batch Normalization)를 사용x
  
NoSkip UNet
```
class UNetDepth_NoSkip(UNetDepthBase):
    def __init__(self, in_channels=3, features=[64,128,256,512]):
        super().__init__(in_channels, features, use_skip=False, use_bn=False)
```
- use_skip=True: 스킵 연결(skip connection)을 사용 x
- use_bn=False: 배치 정규화(Batch Normalization)를 사용x
  
BN UNet
```
class UNetDepth_BN(UNetDepthBase):
    def __init__(self, in_channels=3, features=[64,128,256,512]):
        super().__init__(in_channels, features, use_skip=True, use_bn=True)
```
- use_skip=True: 스킵 연결(skip connection)을 사용 
- use_bn=False: 배치 정규화(Batch Normalization)를 사용

## 4. 실험 결과

### 4.1 정량적 평가

<img src="https://github.com/jiwoong5/cvfinal2/blob/main/2_%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%85%E1%85%A3%E1%86%BC%E1%84%8C%E1%85%A5%E1%86%A8%E1%84%87%E1%85%B5%E1%84%80%E1%85%AD.png" width="1200" height="600" />

- BatchNorm은 학습을 안정화하고 일반화 성능을 높여, 가장 낮은 MAE와 RMSE를 달성
- Skip Connection은 디코더가 세부적인 공간 정보를 복원하는 데 중요하며, 제거 시 성능이 하락
  
### 4.2 정성적 평가

<img src="https://github.com/jiwoong5/cvfinal2/blob/main/2_6.png" width="1200" height="600" />
- 원본 이미지와 Ground Truth 깊이 맵 대비 각 모델의 예측 깊이 맵 시각화  
- BN 이외 모델에서 깊이맵 출력이 비정상적
- 추후 모델 학습을 epoch 를 늘려 시도 필요성 있음

## 5. 분석 및 고찰

- **Skip Connection**: 깊이 추정에서 중요한 공간 정보가 네트워크 깊은 층까지 전달되는 역할을 하므로, 제거 시 정보 손실이 발생해 성능 저하를 유발  
- **Batch Normalization**: 각 층의 입력 분포를 정규화하여 학습 안정성 및 수렴 속도를 개선하며, 깊은 네트워크에서 과적합 감소 효과 기대  
- **에포크 증가**: BatchNorm 모델에서 200 에포크로 장기 학습 시 RMSE가 크게 감소하여 학습 안정성과 표현력 향상을 확인  

## 6. 결론 및 향후 연구

- 모델 구성 변경에 따라 Mono Depth Estimation 성능에 유의미한 차이가 나타났다.  
- Batch Normalization과 충분한 학습 시간 확보가 깊이 추정의 정확도 향상에 효과적임을 확인했다.  
- 향후 다양한 정규화 기법과 구조 변형(예: Attention, Residual Block) 도입을 통한 추가 성능 개선 연구가 필요하다.

