---
title: "PyTorch 기초 학습 (3) — Transform & 원-핫 인코딩"
date: 2026-09-09
draft: false
tags: ["PyTorch", "딥러닝", "study-log"]
categories: ["PyTorch"]
---

> PyTorch 공식 튜토리얼을 학습하며 정리한 내용입니다.
> (https://docs.pytorch.org/tutorials/index.html)

---

## `transform`과 `target_transform`

"모든 TorchVision 데이터셋들은 변형 로직을 갖는 호출 가능한 객체(callable)를 받는 매개변수 두 개(`transform`, `target_transform`)를 갖습니다"라는 설명을 처음 봤을 때, 이게 결국 "transform을 각각 지원한다"는 말인지 궁금했다.

정리하면 세 가지를 말하는 문장이었다.

1. 모든 TorchVision 데이터셋은 `transform`(특징용), `target_transform`(라벨용)이라는 공통 파라미터를 가진다.
2. 여기 넣는 값은 **호출 가능한 객체(callable)** 여야 한다.
3. `torchvision.transforms`에 자주 쓰는 변환 도구(`ToTensor`, `Resize`, `Lambda` 등)가 이미 준비되어 있다.

| 파라미터 | 적용 대상 |
|---|---|
| `transform` | 입력 데이터(feature) |
| `target_transform` | 정답(label) |

#### 오해했던 점

`self.transform`을 하면 어떻게 변환이 되는지 궁금했다. 게다가 스스로 변환 방식을 지정한 적이 없는데 어떻게 작동하는 건지 이해가 안 됐다.

#### 실제로는

`self.transform`은 그냥 변수이고, `Dataset` 객체를 생성할 때 넘겨준 값이 저장되는 것이다.

```python
class MyDataset(Dataset):
    def __init__(self, ..., transform=None):
        self.transform = transform  # 여기서 저장

    def __getitem__(self, idx):
        if self.transform:
            image = self.transform(image)  # 실제 사용
```

```python
training_data = datasets.FashionMNIST (..., transform=ToTensor())  # 여기서 지정한 것
```

이미 생성 시점에 지정을 한 것이고, `__getitem__`은 그것을 나중에 실행만 하는 것이었다.

---

## 원-핫 인코딩(One-hot encoding)과 `Lambda`

#### 오해했던 점

`Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))`코드에서 부분 개념 및 내용은 이해하고 있지만, 전체적인 코드 해석이 어려웠다.

#### 실제로는

- `Lambda` — 커스텀 변환 로직을 직접 정의할 수 있게 해주는 torchvision 클래스
- `torch.zeros(10, dtype=torch.float)` — 크기 10짜리 0벡터 생성 (카테고리 10개용 틀)
- (신규) `.scatter_(0, torch.tensor(y), value=1)` — `dim=0` 기준으로 `y`번째 위치에 값 1을 넣는다

```python
y = 7
torch.zeros(10).scatter_(0, torch.tensor(7), value=1)
# [0,0,0,0,0,0,0,1,0,0] ← 7번째 자리만 1
```

정수 라벨을 **원-핫 인코딩(One-hot encoding)** 벡터로 변환하는 코드였다.

{{< details summary="**원-핫 인코딩에 대한 간단한 예시!**" >}}
카테고리를 "정답 자리만 1, 나머지는 0"인 벡터로 표현하는 방법이다.

```
카테고리: 티셔츠(0), 바지(1), 스니커즈(2) — 총 3종류

티셔츠   → [1, 0, 0]
바지     → [0, 1, 0]
스니커즈 → [0, 0, 1]
```

정수(0, 1, 2)만 쓰면 "스니커즈가 티셔츠보다 크다"처럼 실제로는 없는 순서/크기 관계가 생길 수 있는데, 원-핫 벡터는 각 카테고리를 완전히 독립된 자리로 나눠서 그런 오해를 막아준다.
{{< /details >}}

여기서 `dim=1`이면 어떻게 되는지, 그리고 `y`는 어디서 나온 값인지도 궁금했다.

**`dim=1`의 경우**: 1차원 벡터에서는 `dim=0`만 가능하다. `dim=1`은 2차원 이상일 때 의미가 생긴다.

```python
target = torch.zeros(3, 10)          # 배치 3개, 각 10칸
labels = torch.tensor([[2],[7],[4]])
target.scatter_(1, labels, value=1)  # 각 행마다, 열 방향 지정 위치에 1을 넣음
```

**`y`의 출처**: `y`는 람다 함수의 매개변수이며, `target_transform(label)`처럼 함수가 호출되는 시점에 그 인자값이 전달된다. 실제 호출은 `__getitem__` 내부에서 일어난다.

```python
def __getitem__(self, idx):
    label = self.img_labels.iloc[idx, 1]      # 예: label = 7
    if self.target_transform:
        label = self.target_transform(label)   # 여기서 target_transform(7) 호출 → y=7
```

지정을 한 적이 없는데 어떻게 `y=7`처럼 값이 매핑되는지도 헷갈렸다.

**매핑 원리**: 함수 정의와 호출은 별개였다. 정의 시점(`lambda y: ...`)엔 `y`가 "값이 없는 빈 자리"일 뿐이고, 호출하는 쪽에서 넣은 값이 그 자리에 대입된다.

```python
target_transform(7)
# → (lambda y: ...)(7) → y 자리에 7이 대입됨
# → torch.zeros(10).scatter_(0, torch.tensor(7), value=1)
```

즉 "지정을 안 했다"기보다는, `self.target_transform(label)`을 호출하는 순간이 바로 지정하는 순간이라고 이해하면 됐다.

---

## 참고: torchvision 개요

`torch`가 텐서 연산과 신경망 기본 구조를 담당한다면, `torchvision`은 이미지(컴퓨터 비전) 전용 확장 라이브러리다.

| 하위 모듈 | 역할 |
|---|---|
| `torchvision.datasets` | Fashion-MNIST, MNIST, CIFAR10 등 내장 데이터셋 |
| `torchvision.transforms` | `ToTensor`, `Resize`, `Lambda` 등 전처리 도구 |
| `torchvision.models` | ResNet, VGG 등 사전학습된 모델 |

자매 패키지: `torchaudio`(오디오), `torchtext`(텍스트)

---

## 직접 실습해보기 — 커스텀 Dataset + Transform

배운 내용을 직접 확인해보려고 0~99 숫자를 feature로, 짝수/홀수를 label로 하는 간단한 데이터셋을 만들고, feature를 2배로 만드는 transform까지 적용해봤다.

#### Troubleshooting

클래스 작성 중 겪은 문제와 원인을 간단히 정리하면 다음과 같다.

| 문제 | 원인 |
|---|---|
| `AttributeError: no attribute 'transform'` | `__init__`에서 `self.transform = transform` 저장을 누락 |
| `TypeError: cannot unpack non-iterable NoneType` | `__getitem__`에 `return` 누락 |
| 메서드가 인스턴스 메서드로 인식 안 됨 | `self` 매개변수 누락 |

#### 최종 코드

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Lambda

features = torch.arange(100).reshape(100, 1)
labels = features % 2  # 짝수: 0, 홀수: 1

class NumberDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        if self.transform:
            feature = self.transform(feature)
        return feature, label

dataset = NumberDataset(features, labels, transform=Lambda(lambda x: x * 2))

# 개별 인덱싱으로 transform 적용 확인
feature, label = dataset[5]
print(f"{feature} & {label}")

# DataLoader로 배치 순회
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
for i, (batch_features, batch_labels) in enumerate(dataloader):
    print(f"{i+1} {batch_features.shape} {batch_labels.shape}")
    if i == 2:
        break
```

---
