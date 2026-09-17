---
title: "Zero-DCE 코드 분석 (model.py + Myloss.py)"
date: 2026-09-17
tags: ["Paper_Review", "Zero-DCE", "Code_Analysis", "PyTorch", "study-log"]
categories: ["Code_Notes"]
---

> 문법 개념(unsqueeze, nn.Parameter, torch.mean 방향별 차이, pooling 등)은 [[python-pytorch-syntax-notes]] 참고. 여기는 Zero-DCE 코드 자체의 구조와 로직만 정리.

---

## model.py — DCE-Net 구조

### `__init__` (레이어 준비)
```python
number_f = 32
self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True)       # 입력 3(RGB) → 출력 32
self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) # 32 → 32
self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True)
self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True)
self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) # 입력 64! (skip connection 흔적)
self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) # 입력 64
self.e_conv7 = nn.Conv2d(number_f*2,24,3,1,1,bias=True)       # 입력 64 → 출력 24 (최종 알파맵)

self.maxpool = nn.MaxPool2d(...)      # 실제로는 안 쓰임 (forward에 주석처리)
self.upsample = nn.UpsamplingBilinear2d(...)  # 실제로는 안 쓰임
```
- 클래스 이름 `enhance_net_nopool`에서 "nopool"이 이미 힌트. down-sampling을 뺀 구조라 maxpool/upsample은 정의만 되고 실제 forward에서는 주석처리되어 안 쓰임
- e_conv5,6,7의 입력이 64채널인 이유 = forward에서 `torch.cat`으로 두 특징을 이어붙이기 때문 (아래 참고)

### `forward` — 실제 실행 순서

**1단계: CNN 통과 (6번 ReLU + 마지막 Tanh)**
```python
x1 = self.relu(self.e_conv1(x))
x2 = self.relu(self.e_conv2(x1))
x3 = self.relu(self.e_conv3(x2))
x4 = self.relu(self.e_conv4(x3))

x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))   # x3+x4 이어붙여서 64채널
x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))   # x2+x5 이어붙여서 64채널
x_r = F.tanh(self.e_conv7(torch.cat([x1,x6],1)))     # x1+x6 이어붙여서 64채널, Tanh로 마무리
```
- **`torch.cat([a,b], 1)`**: 채널 방향(1번 차원)으로 두 텐서를 이어붙임. 32+32=64채널
- **왜 이어붙이나 (symmetrical concatenation, skip connection)**: 레이어가 깊어질수록 초반의 세밀한 정보(경계, 텍스처)가 희석되기 쉬움. 얕은 층(x1,x2,x3, 원본에 가까운 정보)을 깊은 층(x4,x5,x6, 더 가공된 판단력)과 다시 섞어줘서, "밝기는 조정하되 원래 구조/디테일은 보존"하게 함
- 짝짓는 패턴이 x1↔x6, x2↔x5, x3↔x4로 **대칭적(symmetrical)** 이라 이런 이름이 붙음
- x_r = 최종 CNN 출력, shape은 24채널 (8회 반복 × RGB 3채널)

**2단계: 24채널을 8개(3채널씩)로 쪼갬**
```python
r1,r2,r3,r4,r5,r6,r7,r8 = torch.split(x_r, 3, dim=1)
```

**3단계: LE-curve를 8번 순서대로 적용 (누적 구조)**
```python
x = x + r1*(torch.pow(x,2)-x)
x = x + r2*(torch.pow(x,2)-x)
x = x + r3*(torch.pow(x,2)-x)
enhance_image_1 = x + r4*(torch.pow(x,2)-x)          # 4번째 결과를 별도 저장
x = enhance_image_1 + r5*(torch.pow(enhance_image_1,2)-enhance_image_1)
x = x + r6*(torch.pow(x,2)-x)
x = x + r7*(torch.pow(x,2)-x)
enhance_image = x + r8*(torch.pow(x,2)-x)            # 최종 결과

r = torch.cat([r1,...,r8],1)   # 8개를 다시 하나로 합쳐서 반환 (Illumination Smoothness Loss 계산용으로 추정)
return enhance_image_1, enhance_image, r
```
- 논문 수식 `LE(x)=x+α·x(1-x)`와 코드의 `x+r·(x²-x)`는 `(x²-x) = -x(1-x)`라서 **부호만 반대인 동일한 식** (r이 논문의 -α 역할)
- 매 줄마다 `x`가 이전 결과를 이어받아 갱신 (누적 적용) — 직전 줄의 출력이 다음 줄의 입력
- 4번째 결과(`enhance_image_1`)를 따로 저장하는 이유: 아마 학습 시 손실 계산에 중간 단계 결과도 활용하기 위함으로 추정 (정확한 용도는 lowlight_train.py 확인 필요)
- 반환값 3개: 4단계 중간결과, 8단계 최종결과, 전체 알파맵(r)

---

## Myloss.py — 손실함수들

### L_color (Color Constancy Loss)
```python
mean_rgb = torch.mean(x,[2,3],keepdim=True)  # 채널별 이미지 전체 평균 (공간 방향 평균!)
mr,mg,mb = torch.split(mean_rgb, 1, dim=1)
Drg = torch.pow(mr-mg,2)
Drb = torch.pow(mr-mb,2)
Dgb = torch.pow(mb-mg,2)
k = torch.pow(Drg+Drb+Dgb, 0.5)   # 유클리드 거리 형태
```
- R,G,B 세 채널의 평균끼리 차이를 구해서, 색 균형이 깨지지 않게 강제
- 논문 수식과 거의 동일, 마지막에 제곱근(0.5제곱) 씌우는 것만 구현상 추가

### L_spa (Spatial Consistency Loss) — (개인적으로 가장 이해가 힘들었던...)
```python
# __init__: 4방향 필터 준비 (학습 안 되게 고정, nn.Parameter+requires_grad=False)
kernel_left/right/up/down = [고정된 3x3 필터]
self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)  # 등등
self.pool = nn.AvgPool2d(4)   # 4x4 지역 평균용

# forward(org, enhance):
org_mean = torch.mean(org,1,keepdim=True)      # 채널 평균 (흑백화)
enhance_mean = torch.mean(enhance,1,keepdim=True)
org_pool = self.pool(org_mean)                  # 4x4 지역 평균 (공간 압축)
enhance_pool = self.pool(enhance_mean)

D_org_left = F.conv2d(org_pool, kernel_left, padding=1)   # 각 방향 이웃과의 차이
# ... right, up, down도 동일 (org, enhance 각각)

D_left = (D_org_left - D_enhance_left)²   # 원본에서의 차이 vs 향상본에서의 차이 비교
# ... 4방향 다 계산
E = D_left + D_right + D_up + D_down      # 최종 합산
return E
```
- **핵심 논리**: "이 지역이 이웃보다 얼마나 밝은가"라는 상대적 패턴이 원본↔향상본에서 똑같이 유지되도록 강제 → 전체가 밝아져도 명암 대비(구조)는 보존
- **주의**: 채널 평균(색 합치기)과 AvgPool(위치 합치기)은 완전히 다른 축의 평균이라 헷갈리기 쉬움 — 자세한 건 문법 노트 참고
- `weight_diff`, `E_1` 변수는 계산은 되지만 **최종 return값 E에는 안 들어감 → 사실상 죽은 코드로 보임** (0.3 임계값 기준으로 어두운 곳 가중치 0.5, 밝은 곳 1로 나누는 로직인데 실제 안 쓰임)

### L_exp (Exposure Control Loss)
```python
def __init__(self, patch_size, mean_val):
    self.pool = nn.AvgPool2d(patch_size)  # 16x16
    self.mean_val = mean_val               # 목표 밝기 0.6

def forward(self, x):   # 인자 하나만 받음! (원본 필요 없음)
    x = torch.mean(x,1,keepdim=True)   # 흑백화
    mean = self.pool(x)                 # 16x16 지역 평균
    d = torch.mean(torch.pow(mean - self.mean_val, 2))  # 목표값과의 차이, 전체 평균
    return d
```
- **L_spa와 결정적 차이**: 원본과 비교 안 함. 오직 "향상된 결과가 목표 밝기(0.6)에 얼마나 가까운지"만 체크
- forward(self, x) — 인자가 하나뿐인 게 "원본 비교 없음"의 증거

### L_TV (Illumination Smoothness Loss)
```python
count_h = (h-1) * w   # 세로 방향 이웃쌍 개수 (마지막 줄은 짝이 없어서 h-1)
count_w = h * (w-1)   # 가로 방향 이웃쌍 개수

h_tv = torch.pow((x[:,:,1:,:] - x[:,:,:h-1,:]), 2).sum()  # 세로로 이웃한 칸끼리 차이
w_tv = torch.pow((x[:,:,:,1:] - x[:,:,:,:w-1]), 2).sum()  # 가로로 이웃한 칸끼리 차이

return weight * 2 * (h_tv/count_h + w_tv/count_w) / batch_size
```
- **L_spa와 다른 점**: 원본/향상본 비교가 아니라, **이미지(또는 α맵) 하나 안에서** 이웃 칸끼리 비교
- `x[:,:,1:,:]`(2번째 행부터 끝) vs `x[:,:,:h-1,:]`(1번째부터 마지막-1행) → 세로 이웃끼리 뺄셈
- 실제로는 α맵(A)에 적용됨 → 최종 이미지 자체를 뭉개는 게 아니라, "얼마나 밝힐지"의 패턴만 매끄럽게 함 → L_spa(대비 보존)와 서로 균형을 이뤄 경계가 과하게 뭉개지는 걸 방지
- 없으면 아티팩트(얼룩덜룩함) 발생 (Ablation에서 확인됨)

### Sa_Loss (논문 4개 손실함수엔 없는 추가 클래스)
```python
r,g,b = torch.split(x, 1, dim=1)
mean_rgb = torch.mean(x,[2,3],keepdim=True)  # 채널별 전체 공간 평균
mr,mg,mb = torch.split(mean_rgb, 1, dim=1)
Dr = r-mr; Dg = g-mg; Db = b-mb   # 각 픽셀 - 그 채널의 전체 평균
k = torch.pow(Dr²+Db²+Dg², 0.5)  # 유클리드 거리
k = torch.mean(k)
```
- **핵심**: "각 픽셀값이 이미지 전체 평균(그 채널의)에서 얼마나 튀는지"를 측정 → 채도(색의 다채로움 정도)를 수치화하는 것으로 추정
- 논문 본문 4개 손실함수엔 없음 → 실험 흔적일 가능성 높음, 실제 학습 사용 여부는 lowlight_train.py 확인 필요

### perception_loss (VGG16 기반, 논문에 없는 추가 클래스)
```python
features = vgg16(pretrained=True).features   # ImageNet으로 이미 학습된 VGG16의 특징추출부만 가져옴
# VGG16 레이어를 4구간으로 잘라서 각각 nn.Sequential 상자에 담음
self.to_relu_1_2 = [features[0]~[3]]   # 4개 레이어
self.to_relu_2_2 = [features[4]~[8]]   # 5개
self.to_relu_3_3 = [features[9]~[15]]  # 7개
self.to_relu_4_3 = [features[16]~[22]] # 7개
# 전부 requires_grad=False (VGG16 자체는 학습 안 시킴, 이미 완성된 도구로만 씀)

def forward(self, x):
    h = self.to_relu_1_2(x); h_relu_1_2 = h
    h = self.to_relu_2_2(h); h_relu_2_2 = h   # 이전 단계 결과를 다음 입력으로 (이어달리기)
    h = self.to_relu_3_3(h); h_relu_3_3 = h
    h = self.to_relu_4_3(h); h_relu_4_3 = h
    return h_relu_4_3   # 가장 깊은 단계 특징만 반환 (중간단계 4개 다 뽑아놓고 마지막것만 씀)
```
- **VGG16이란**: ImageNet(수백만 장 이미지, 1000개 카테고리)으로 이미 학습된 유명 CNN. "이미지를 잘 이해하는 안목"을 가진 모델
- **여기서 쓰는 이유(Perceptual Loss 개념)**: 픽셀값을 직접 비교하는 대신, "VGG16이라는 이미 훈련된 눈으로 봤을 때 두 이미지(원본 vs 향상본)가 얼마나 비슷하게 인식되는지"를 비교하려는 것. 사람이 느끼는 유사성을 더 잘 반영한다고 알려진 방식
- **주의**: 이 forward 자체는 "이미지 1장 → 512채널짜리 특징 텐서 1개"를 반환할 뿐, 아직 손실값(숫자)이 아님. 원본용/향상본용 두 번 호출해서 그 둘을 빼고 제곱해야 비로소 손실값이 나옴 — 그 비교 코드는 이 파일에 없고 lowlight_train.py에 있을 것으로 추정
- VGG16을 통째로 가져오는 건 무거운 작업이라, Zero-DCE의 "경량화" 강점과는 안 맞음 → 최종 학습엔 안 쓰였을 가능성 높은 실험용 코드로 추정

---

## lowlight_train.py — 학습 루프 조립

지금까지 뜯어본 model.py(DCE-Net)와 Myloss.py(4개 손실함수)가 실제로 어떻게 하나의 학습 과정으로 엮이는지.

### 전체 흐름
```
설정값 파싱 (argparse)
  → 모델 생성 + 초기화 (DCE_net = model.enhance_net_nopool())
  → 데이터 로더 준비 (glob으로 파일 목록 수집 + shuffle)
  → 손실 함수 4개 인스턴스 생성 (L_color, L_spa, L_exp, L_TV)
  → optimizer 생성 (Adam, DCE_net.parameters() 전달)
  → 에폭 반복:
      배치 입력 → DCE_net(x)로 예측 (enhance_image_1, enhance_image, r 반환)
      → 각 손실함수 계산 → 총 손실(loss) 합산
      → optimizer.zero_grad() → loss.backward() → clip_grad_norm → optimizer.step()
      → 주기적으로 로그 출력 및 가중치(.pth) 저장
```

### 모델 출력 3개(enhance_image_1, enhance_image, r) 중 실제 test에서 쓰이는 것
```python
_, enhanced_image, _ = DCE_net(data_lowlight)
```
- `lowlight_test.py`에서는 이렇게 **가운데 값(최종 8단계 결과, enhance_image)만** 이름 붙여 쓰고, 나머지(4단계 중간결과 enhance_image_1, 전체 알파맵 r)는 `_`로 버림
- → "왜 model.py가 이미지를 두 개나 반환하나"에 대한 답 일부: **테스트 단계에서는 최종 결과 하나만 쓰면 충분**하고, 나머지 반환값은 (아마도) 학습 단계의 손실 계산에서만 쓰이는 것으로 추정됨. 학습 루프 안에서 `enhance_image_1`이 정확히 어느 손실에 쓰이는지는 아직 확인 필요

### optimizer 설정 (실제 코드)
```python
optimizer = torch.optim.Adam(DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
```
- `DCE_net.parameters()`: DCE-Net 안의 모든 conv 레이어 가중치(e_conv1~7)를 옵티마이저에게 넘김 — 이게 학습 대상. (L_spa의 방향 필터처럼 `requires_grad=False`인 것들은 여기 안 잡힘)