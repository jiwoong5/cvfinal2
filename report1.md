# 원본 StereoNet vs 개선된 ImprovedStereoNet 구현 차이 및 성능 비교 보고서

## 1. 서론 (Introduction)
steronet과 실험목표
- StereoNet은 두 입력 이미지(좌우 시점)로부터 각 픽셀에 대한 깊이 정보를 추정하는 경량화된 CNN 기반 스테레오 매칭 모델이다.
- 본 프로젝트에서는 StereoNet의 **Feature Extractor 모듈을 개선**하여 **표현력 향상** 및 **정확도 개선**을 도모하였다.
- 본 보고서에서는 기존 구조와 개선 구조를 비교하여 구조적 차이, 기대 효과, 그리고 성능에 미치는 영향을 설명한다.

stereonet 기본 구조
- Conv2D → BatchNorm → ReLU 구조의 단순 반복
- 총 5개의 계층, 모두 32채널
- 잔차 연결(residual connection) 없음

Gradient Vanishing 문제란?
- 딥러닝 학습에서 **계층이 깊어질수록 역전파 시 기울기(gradient)가 점점 작아짐**
- 특히 ReLU, BatchNorm 등 비선형성/정규화가 반복되며 정보 손실이 누적
- 결과적으로 앞쪽 계층(입력층 근처)은 거의 학습이 이루어지지 않음 → **표현력 제한**

Gradient Vanishing 문제 원인 분석
| 원인 | 설명 |
|------|------|
| 깊은 신경망 | 5층 이상의 CNN 구조에서는 역전파 시 기울기 소실 문제가 빈번하게 발생함 |
| Residual 연결 없음 | 이전 출력을 현재 출력에 더해주는 skip connection이 없으면 학습 정보가 누락됨 |
| 반복된 ReLU, BN | ReLU는 음수를 0으로, BN은 분포를 정규화시켜, 점진적으로 정보가 사라지는 현상 초래 |

## 2. 핵심 아이디어: Residual Learning
- 일반적인 깊은 신경망에서는 네트워크의 깊이가 증가함에 따라 학습이 어려워지고, 오히려 성능이 감소하는 문제가 발생합니다.
- 이를 해결하기 위해 ResNet 논문에서는 Residual Learning 구조를 도입하였습니다.
- 즉, 전체 함수를 직접 학습하기보다, 입력 대비 변화량(잔차)만 학습하도록 유도합니다.
- 출처: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

수식표현: 
- H(x)=F(x)+x
- H(x): 우리가 최종적으로 원하는 출력
- x: 입력
- F(x) = H(x)−x: 학습할 잔차(residual) 함수

해석: 
- 일반적으로 H(x)를 직접 학습하는 것은 어려울 수 있음.
- 하지만 F(x) (즉, 입력에 더해야 할 변화량)를 학습하는 것은 더 쉽고 안정적임.
- 이 구조 덕분에 역전파 시 gradient가 잘 흐를 수 있게 되어, gradient vanishing 문제가 완화됨.

결과:
- 매우 깊은 네트워크(예: ResNet-152)도 안정적으로 학습 가능
- 기존 네트워크보다 빠른 수렴 속도와 높은 정확도 달성

## 3. 모델 아키텍쳐 소개
ResidualBlock 구현
```
python
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        ...
        self.down = nn.Identity() if 조건 else nn.Conv2d(...)
    
    def forward(self, x):
        identity = self.down(x)
        out = self.conv_bn_relu_block(x)
        return self.relu(out + identity)
```
개선된 구조: ResidualBlock 도입
| 항목                 | 기존 FeatureExtractor                    | 개선된 ImprovedFeatureExtractor       |
|----------------------|------------------------------------------|----------------------------------------|
| 계층 구성            | Conv-BN-ReLU 반복                         | Conv-BN-ReLU + ResidualBlock 4개      |
| Skip Connection 사용 | ❌ 없음                                   | ✅ 있음 (Residual 연결 도입)           |
| Gradient 흐름        | 계층 깊어질수록 소실될 수 있음            | 우회 경로 확보로 gradient 흐름 원활   |
| 학습 안정성          | 초기 수렴 속도 느리고 불안정할 수 있음    | 더 빠르고 안정적인 수렴                |
| 표현력               | 단순 패턴 학습에 그칠 수 있음             | 더 복잡한 특성 추출 가능               |
| 성능 기대            | 제한적 (MAE 향상 여지 적음)               | 향상된 성능 기대 (MAE 감소 가능성 높음) |

## 4. 실험 조건

| 실험 번호 | Feature Extractor 종류 | 에포크(Epoch) 수 |
|-----------|------------------------|------------------|
| 1         | 기본 Feature Extractor | 20               |
| 2         | 개선된 Feature Extractor | 20               |
| 3         | 개선된 Feature Extractor | 200              |

## 5. 실험 결과
### 1. 정량적 비교
![1_5](https://github.com/jiwoong5/cvfinal2/blob/main/1_5.png)
| 모델                        | RMSE    | 기존 모델 대비 개선율(%)        |
|-----------------------------|---------|-------------------------------|
| Original (기본 Feature, 20 epoch)     | 1.9388  | -                             |
| Improved (개선된 Feature, 20 epoch)   | 1.7450  | 약 10.0% 개선                  |
| Improved (개선된 Feature, 200 epoch)  | 1.3569  | 약 30.02% 개선                  |

### 2. 정성적 비교

<img src="https://github.com/jiwoong5/cvfinal2/blob/main/1_7_testing.png" width="800" height="600" />

- 학습에 이용되지 않은 testing set 을 이용한 시각화
- 기본 모델의 경우 좁은 영역을 깊게 예측하고, 개선 모델의 경우 넓은 영역을 덜 깊게 예측

<img src="https://github.com/jiwoong5/cvfinal2/blob/main/1_7_training.png" width="800" height="600" />

- 학습에 이용된 training set 을 이용한 시각화
- gt와의 밝기 비교를 통해 전반적으로 개선된 모델이 좋은 성능을 낸다는 것을 알 수 있음

## 6. 결론 및 향후 과제

결과 요약:
- 본 실험에서는 기본 StereoNet 구조와 비교하여, Residual Block을 도입한 ImprovedStereoNet이 정량적(RMSE 감소) 및 정성적(오차 시각화) 측면에서 확실한 성능 개선을 이끌어내는 것을 확인하였다. 
- 특히, 장기 학습(200 epoch)을 통해 기존 대비 30% 이상의 RMSE 개선을 달성하며, 개선된 Feature Extractor의 학습 효율성과 일반화 능력을 입증하였다.

한계점:
- 실험에 사용된 네트워크 구조는 여전히 간단한 CNN 기반으로, 복잡한 장면이나 얇은 구조물에 대한 정밀한 깊이 추정은 어렵다.
- Cost Volume Aggregation 또한 정교한 정규화나 attention 메커니즘을 활용하지 않았다.

향후 연구 방향:
- 보다 정교한 feature extractor 설계 (예: attention-based feature, multi-scale feature fusion 등).
- 3D Convolution 기반의 cost volume regularization 구조로 확장.
- 모델 경량화를 통한 실시간 응용 가능성 평가.

결론:
- 본 실험은 Residual 구조의 도입이 깊이 추정 모델의 안정성과 성능에 실질적인 영향을 준다는 점을 실증적으로 보여주며, 향후 Stereo Vision 기반 응용에서 효과적인 Feature Extractor 설계의 중요성을 부각시킨다.
