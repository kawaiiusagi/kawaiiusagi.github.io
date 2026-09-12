---
title: "PyTorch 기초 학습 (7) — 모델 저장하고 불러오기"
date: 2026-09-12
draft: false
tags: ["PyTorch", "딥러닝", "study-log"]
categories: ["PyTorch"]
---

> PyTorch 공식 튜토리얼을 학습하며 정리한 내용입니다.
> (https://tutorials.pytorch.kr/beginner/basics/saveloadrun_tutorial.html)

---

## 모델 가중치 저장하고 불러오기

PyTorch 모델은 학습한 매개변수를 `state_dict`라고 불리는 내부 상태 사전(internal state dictionary)에 저장한다.

내부 상태 사전은 **"레이어 이름: 그 레이어의 현재 가중치 텐서"** 형태로 정리된 평범한 파이썬 딕셔너리이다.

```python
model.state_dict()
```

```python
OrderedDict([
    ('linear_relu_stack.0.weight', tensor([[...]])),
    ('linear_relu_stack.0.bias', tensor([...])),
    ('linear_relu_stack.2.weight', tensor([[...]])),
    ('linear_relu_stack.2.bias', tensor([...])),
    ('linear_relu_stack.4.weight', tensor([[...]])),
    ('linear_relu_stack.4.bias', tensor([...])),
])
```

- **상태(state)**: 모델이 지금 갖고 있는 모든 가중치 값들 (학습이 진행될수록 계속 바뀜)
- **사전(dictionary)**: "레이어 이름: 값" 형태로 정리된 구조
- **내부(internal)**: 파이토치가 모델 안에서 자동으로 관리하고 있는 정보

`nn.Sequential` 안의 레이어들은 0번, 1번, 2번... 순서로 번호가 매겨지는데, `ReLU`처럼 가중치가 없는 레이어는 `state_dict`에 아예 나타나지 않고, 가중치를 가진 `nn.Linear`만 `linear_relu_stack.0`, `linear_relu_stack.2`, `linear_relu_stack.4`처럼 키로 등장하게 된다.

각 값(value)은 "학습을 통해 지금까지 계산된, 실제 숫자로 채워진 텐서"다. 즉 **"어느 레이어인지(키)에 대해, 그 레이어가 지금 갖고 있는 실제 가중치 값(텐서)"** 이 짝지어진 구조였다.

이 상태 값들은 `torch.save`로 저장할 수 있다.

```python
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')
```

모델 가중치를 불러오려면, 먼저 동일한 모델의 인스턴스를 생성한 다음 `load_state_dict()`로 매개변수들을 불러온다.

```python
model = models.vgg16()  # weights를 지정하지 않았으므로, 학습되지 않은 모델을 생성함
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
```

**코드**
```python
import torch
import torchvision.models as models

# 1. 사전학습된 VGG16 모델 불러오기
model = models.vgg16(weights='IMAGENET1K_V1')

# 2. 모델 가중치(state_dict)만 저장
torch.save(model.state_dict(), 'model_weights.pth')

# 3. 동일한 구조의 새 모델 인스턴스 생성 (가중치는 없는 상태)
model = models.vgg16()  # weights를 지정하지 않았으므로, 학습되지 않은 모델을 생성함

# 4. 저장해둔 가중치를 불러와서 채워넣기
model.load_state_dict(torch.load('model_weights.pth'))

# 5. 평가 모드로 전환
model.eval()

# 6. 모델 구조 출력 (여기서 print(model)과 동일하게 VGG(...) 구조가 출력됨)
print(model)
```
**출력 일부**
```plain_text
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    ..... 
```

이 구조를 보면 `Conv2d`(합성곱층), `ReLU`(활성화 함수), `MaxPool2d`, `Linear`(완전연결층), `Dropout`이 실제 유명 모델(VGG16)에서 어떻게 조합되어 쓰이는지 확인할 수 있었다. 

> **참고**: 추론(inference)을 하기 전에 `model.eval()`을 호출해서 드롭아웃과 배치 정규화를 평가 모드로 설정해야 한다. 그렇지 않으면 일관성 없는 추론 결과가 생성된다.

---

## `train()`/`eval()`을 구분해야 하는 이유

#### 오해했던 점

`model.train()`과 `model.eval()`이 왜 필요한지, 그리고 왜 지금 만든 모델(`NeuralNetwork`)에는 "없어도 된다"고 하는지 헷갈렸다.

#### 실제로는

`Dropout`이나 `BatchNorm` 같은 레이어는 **학습 중과 평가 중에 동작 방식 자체가 달라진다.**

- `Dropout`: 학습 중에는 과적합 방지를 위해 뉴런의 일부를 랜덤하게 꺼버리지만, 평가할 때는 예측이 일관되게 나와야 하니 전부 다 사용한다.
- `BatchNorm`: 학습 중에는 현재 배치의 평균/분산을 이용해 정규화하지만, 평가할 때는 학습 중 저장해둔 전체 평균/분산을 대신 사용한다.

`model.train()`/`model.eval()`은 이런 레이어들에게 "지금이 학습 중인지 평가 중인지" 알려주는 스위치다. 

지금까지 만든 `NeuralNetwork`는 `Linear`와 `ReLU`만 있어서 이 스위치의 영향을 안 받지만, VGG16처럼 `Dropout`이 들어간 모델에서는 이 구분을 빼먹으면 같은 입력을 넣어도 예측 결과가 매번 달라지는 문제가 생긴다. 

그래서 습관적으로 항상 구분해서 쓰는 게 좋은 습관이라는 걸 확인했다.

---

## 모델의 형태를 포함하여 저장하고 불러오기

모델의 가중치를 불러올 때는 신경망의 구조를 정의하기 위해 모델 클래스를 먼저 생성해야 했다. 이 클래스의 구조 자체를 모델과 함께 저장하고 싶으면, (`model.state_dict()`가 아닌) `model` 자체를 저장 함수에 전달한다.

```python
torch.save(model, 'model.pth')
```

다음과 같이 모델을 불러올 수 있다.

```python
model = torch.load('model.pth')
```

> **참고**: 이 접근 방식은 파이썬 `pickle` 모듈을 사용해서 모델을 직렬화(serialize)하기 때문에, 모델을 불러올 때 실제 클래스 정의(definition)에 의존한다. 

최신 PyTorch(2.6 이상)에서는 보안 강화를 위해 `torch.load`의 `weights_only` 기본값이 `True`로 바뀌어서, 이 방식으로 저장한 파일을 그대로 불러오면 아래와 같은 에러가 날 수 있다고 한다.

```
_pickle.UnpicklingError: Weights only load failed.

WeightsUnpickler error: Unsupported global: GLOBAL torchvision.models.vgg.VGG was not an allowed global by default.
Please use `torch.serialization.add_safe_globals([torchvision.models.vgg.VGG])` 
or the `torch.serialization.safe_globals([torchvision.models.vgg.VGG])` context manager to allowlist this global if you trust this class/function.
```

신뢰할 수 있는 출처의 파일이 확실할 때만 `weights_only=False`로 불러오거나, `add_safe_globals`로 허용 목록에 추가해야 한다는 점이 최신 파이토치 버전의 유의사항이라고 한다.

---

## `state_dict` vs `model` 통째로 저장 — 정리

| 저장 방식 | 저장되는 것 | 불러올 때 필요한 것 |
|---|---|---|
| `torch.save(model.state_dict(), ...)` | 가중치 값들만 | 동일한 모델 클래스를 먼저 만들어야 함 |
| `torch.save(model, ...)` | 모델 구조 + 가중치 값 전부 | 클래스 정의가 그대로 있어야 함 (pickle 방식) |

**비유하면**: `state_dict`만 저장하는 건 "게임 세이브 파일"만 저장하는 것과 같아서, 게임 캐릭터의 설계도(모델 클래스 코드)는 따로 갖고 있어야 불러올 수 있다. 

`model` 자체를 저장하면 캐릭터 설계도까지 통째로 저장하는 셈이라 편리하지만, pickle 방식의 보안 이슈가 있어 최신 버전에서는 더 주의가 필요하다.

---

*이전 편: 6편 모델 매개변수 최적화하기*
