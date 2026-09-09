---
title: "PyTorch 기초 학습 (2) — Dataset & DataLoader"
date: 2026-09-09
draft: false
tags: ["PyTorch", "딥러닝", "study-log"]
categories: ["PyTorch"]
---

> PyTorch 공식 튜토리얼을 학습하며 정리한 내용입니다.
> (https://docs.pytorch.org/tutorials/index.html)

---

## 왜 데이터셋 코드와 학습 코드를 분리하는가

데이터 준비 코드(파일 읽기, 전처리)와 모델 학습 코드(forward, loss, backward)를 한 덩어리로 섞지 말고 분리하는 것이 이상적이라고 한다. 이유는 두 가지다.

- **가독성**: 어디가 데이터 처리이고 어디가 학습 로직인지 명확하게 구분된다.
- **모듈성**: 재사용이 가능하고, 서로 독립적으로 수정할 수 있다.

(참고: train/test split과는 다른 개념이다. 여기서 말하는 건 "구조적 분리"다.)

---

## `Dataset` vs `DataLoader`

| | `Dataset` | `DataLoader` |
|---|---|---|
| 역할 | 데이터 1개를 어떻게 가져올지 정의 | Dataset을 배치/셔플/병렬로 감싸서 꺼내줌 |
| 사용 방식 | `dataset[i]` | `for batch in dataloader:` |
| 필수 메서드 | `__len__`, `__getitem__` | 없음 (Dataset을 받아서 씀) |

```python
from torch.utils.data import Dataset, DataLoader

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
```

#### 오해했던 점

`class NumberDataset(Dataset):`이 상속 문법이라는 건 알고 있었지만, 이 상속이 실제로 어떤 역할을 하는지는 명확하지 않았다. 단순히 문법을 따라 쓴 것뿐인지, 아니면 실제로 뭔가 작동에 영향을 주는 건지 궁금했다.

#### 실제로는

상속은 단순한 형식이 아니라, `DataLoader`가 그 객체를 신뢰하고 사용하는 근거로 작동한다. 

`class NumberDataset(Dataset):`이라고 쓰면, `Dataset`이 정한 규칙(`__len__`, `__getitem__`을 반드시 구현해야 한다는 규칙)을 물려받게 된다. 

`DataLoader`는 이 상속 관계를 보고 "이 객체는 Dataset 규격을 따르는구나"라고 판단하고, 내부에서 `len(dataset)`이나 `dataset[i]` 같은 호출을 안전하게 수행할 수 있다. 

즉 상속은 `DataLoader`가 `Dataset`을 다룰 때 "이 객체가 이런 메서드들을 갖고 있을 것"이라고 믿을 수 있게 해주는 일종의 계약 역할을 한다.

---

## 파일 기반 `__getitem__` 실전 예시

```python
def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
    image = read_image(img_path)
    label = self.img_labels.iloc[idx, 1]
    if self.transform:
        image = self.transform(image)
    if self.target_transform:
        label = self.target_transform(label)
    sample = {"image": image, "label": label}
    return sample
```

- `self.img_labels.iloc[idx, 0]` — CSV에서 idx번째 행, 0번째 열(파일명)을 가져온다.
- `os.path.join(...)` — 폴더 경로와 파일명으로 전체 경로를 만든다.
- `read_image(img_path)` — 이미지를 텐서로 불러온다.
- 이후 transform/target_transform을 적용하고 딕셔너리로 반환한다.

#### 오해했던 점

`loc`이 위치 기준이고 `iloc`이 번째(순서) 기준이라고 알고 있었다.

#### 실제로는

반대였다.
- **`loc`** = **이름(라벨)** 기준 (`df.loc[10]` → 인덱스 라벨이 10인 행)
- **`iloc`** = **번째(정수 위치)** 기준 (`df.iloc[0]` → 0번째 순서, i = integer)

`sample = {"image": image, "label": label}` 줄은 처리 끝난 image와 label을 딕셔너리 하나로 묶는 것이다. 튜플 리턴(`return image, label`)과 비교하면, 딕셔너리는 키 이름으로 명시적으로 꺼낼 수 있어 데이터 종류가 늘어나도 헷갈리지 않는 장점이 있다.

---

## DataLoader 순회 — `for` vs `next(iter())`

#### 오해했던 점

`train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)`라는 줄 자체가 반복문인 줄 알았다.

#### 실제로는

이 줄 자체는 반복문이 아니다. 반복 가능한 객체(iterable)를 만드는 코드일 뿐이다.

```python
train_dataloader = DataLoader(...)  # 객체만 생성 (자판기 설치)
for batch in train_dataloader:      # 실제 반복은 여기서 발생 (버튼 누르기)
```

`next(iter(...))`가 실제로 어떻게 작동하는지도 헷갈렸는데, 단계를 나눠보니 명확해졌다.

1. `train_dataloader`는 iterable(반복 가능한 잠재력만 있는 객체)이다.
2. `iter(train_dataloader)` → 이터레이터 생성 (현재 위치를 기억하는 객체, 일종의 "책갈피")
3. `next(iterator)` → 다음 배치 하나를 꺼내고 위치를 전진시킨다.

내부적으로는 던더 메서드 `__iter__`(→ `iter()`가 호출), `__next__`(→ `next()`가 호출)로 구현되어 있다.

```python
for batch in train_dataloader:
    ...
# 사실상 동일:
iterator = iter(train_dataloader)
while True:
    try:
        batch = next(iterator)
    except StopIteration:
        break
```

이터레이터가 순차적으로 위치를 기억하며 호출해주는 거라면, `iter`에 `next`를 씌워서 자동으로 계속 순차 호출되는 거라고 생각했는데, 그건 아니었다. 

`next(iter(...))` 한 줄은 딱 한 번만 실행되고 끝난다. 자동 반복 기능은 없으며, 여러 번 부르려면 직접 여러 번 호출해야 한다. 

자동으로 계속 반복해주는 건 `for`문이고, `next(iter(...))`는 그 과정 중 한 스텝만 수동으로 재현한 것이다.

또한 `train_features[0]`에서 `[0]`은 배치 하나(64장)를 뽑고 그중 0번째 장을 뽑는 것을 의미한다.

```python
train_features.shape       # [64, 1, 28, 28]
train_features[0].shape    # [1, 28, 28] ← 1장만 남음
```

`for i, (batch_features, batch_labels) in enumerate(dataloader):`에서 `batch_features`라는 변수를 따로 만든 적이 없는데 알아서 그렇게 되는 것도 궁금했는데, `batch_features`, `batch_labels`는 파이토치가 정한 고정 이름이 아니라 작성자가 자유롭게 지은 변수명이었다. 

`Dataset.__getitem__`이 `(feature, label)` 튜플을 리턴하므로 `DataLoader`도 그 순서대로 튜플을 넘겨주는 것이고, 파이썬 언패킹은 순서만 지키면 이름은 자유다.

`enumerate` + `for`문 대신 `next()`를 반복문에서 직접 쓰는 것도 가능한지 확인해봤다.

```python
# 방식 1: for + enumerate (실무 학습 루프에서 사용)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
for i, (batch_features, batch_labels) in enumerate(dataloader):
    print(f"{i+1} {batch_features.shape} {batch_labels.shape}")
    if i == 2:
        break
```

```python
# 방식 2: next()를 직접 반복 호출 (동일한 결과)
iterator = iter(dataloader)
for i in range(3):
    batch_features, batch_labels = next(iterator)
    print(f"{i+1} {batch_features.shape} {batch_labels.shape}")
```

```python
# 방식 3: while + next() + StopIteration 직접 처리 (끝까지 순회)
iterator = iter(dataloader)
while True:
    try:
        batch_features, batch_labels = next(iterator)
        print(batch_features.shape)
    except StopIteration:
        break
```

| | 가능 여부 | 특징 |
|---|---|---|
| `for` 문 | 가능 | `StopIteration` 자동 처리, 간결함 — 실무 표준 |
| `next()` + `range`/`while` | 가능 | 되지만 종료 시점을 직접 챙겨야 함 |

세 방식 다 결과는 같지만, `next(iter(...))` 단독 사용은 "배치 하나만 잠깐 확인"하는 디버깅 용도가 가장 자연스럽다는 걸 알게 됐다.

---

*다음 편: 3편 Transform & 원-핫 인코딩*
