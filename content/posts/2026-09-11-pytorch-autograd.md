---
title: "PyTorch 기초 학습 (5) — torch.autograd를 사용한 자동 미분"
date: 2026-09-11
draft: false
tags: ["PyTorch", "딥러닝", "study-log"]
categories: ["PyTorch"]
---

> PyTorch 공식 튜토리얼을 학습하며 정리한 내용입니다.
> (https://tutorials.pytorch.kr/beginner/basics/autogradqs_tutorial.html)

---

## 역전파(Backpropagation)란

신경망을 학습할 때 가장 자주 쓰이는 알고리즘이 **역전파**라고 하는데, 이게 정확히 뭘 하는 건지 궁금했다.

역전파는 **"얼마나 틀렸는지를 바탕으로, 각 레이어의 가중치를 어떻게 조정해야 할지 계산하는 방법"** 이다. 지금까지 배운 `forward`(정방향)와 정반대 방향으로 진행된다.

```python
logits = model(X)   # forward: 입력 → 예측값 (정방향)
```

역전파는 예측이 끝난 뒤, 예측이 정답과 얼마나 다른지(오차)를 계산하고, 그 오차를 거꾸로 흘려보내면서 각 레이어가 얼마나 잘못했는지 알아내는 과정이다.

```
정방향(forward):  입력 → Linear → ReLU → Linear → ReLU → Linear → 출력
                                                                      ↓
                                                              (정답과 비교, 오차 계산)
                                                                      ↓
역방향(backward): 입력 ← Linear ← ReLU ← Linear ← ReLU ← Linear ← 오차
```

#### 오해했던 점

가중치를 "찾는" 게 목적이면, 정방향으로 계산해야지 왜 거꾸로(역방향) 계산하는지 이해가 안 됐다.

#### 실제로는

역전파가 계산하는 건 "가중치 값 자체"가 아니라, **"가중치를 얼마나 바꿔야 하는지에 대한 방향과 크기(기울기)"** 였다. 신경망은 여러 레이어가 겹겹이 쌓인 함수라서, 앞쪽 레이어의 가중치가 최종 오차에 미친 영향을 알려면 뒤쪽 레이어들을 다 거쳐야 하는 구조이기 때문이다.(연쇄법칙, chain rule).

```
오차 = f(레이어3(레이어2(레이어1(입력))))
```

레이어1의 가중치 변화가 오차에 미치는 영향을 계산하려면, 오차 쪽에서부터 시작해서 각 레이어의 영향을 하나씩 곱해가며 앞으로(입력 방향으로) 전달해야 한다. 그래서 반드시 **결과 쪽에서 거슬러 올라가야만 계산이 가능**하고, 이게 "역"전파라고 부르는 이유였다.

정리하면 "3번, 2번 레이어의 계산 결과를 재사용해서, 연쇄법칙으로 1번 레이어의 기울기를 정확히 계산해낸다"는 흐름이라고 생각하면 된다. 뒤에서 계산한 값을 다음 계산에 그대로 곱해서 넘겨주는 구조라 "전파(propagation)"이고, 그 방향이 출력→입력이라 "역(back)"이 붙어 역전파라고 부른다.

---

## 예제 코드로 보는 연산 그래프

```python
import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
```

이 코드에서 `w`와 `b`가 최적화해야 하는 매개변수이고, 이 값들에 대한 손실 함수의 변화도(gradient)를 계산할 수 있어야 해서 `requires_grad`를 설정한다.

---

## `requires_grad`의 역할

#### 오해했던 점

`w = torch.randn(5, 3, requires_grad=True)`에서 `requires_grad`가 정확히 뭘 하는 옵션인지 몰랐다.

#### 실제로는

`requires_grad=True`는 **"이 텐서에 대한 연산 과정을 기록해서, 나중에 기울기를 자동으로 계산할 수 있게 준비해두라"** 는 옵션이었다.

```python
w = torch.randn(5, 3)   # requires_grad 없음 (기본값 False)
y = w * 2
y.sum().backward()   # 에러남! w에 대한 기울기를 계산할 수 없음
```

`requires_grad=True`인 텐서는 이 텐서가 관여한 모든 연산을 계속 추적해서, `.backward()`를 호출하면 그 기록을 거슬러 올라가며 기울기를 계산할 수 있다. 학습을 통해 값이 바뀌어야 하는 것들(가중치, 편향)은 반드시 이 옵션이 켜져 있어야 한다.

> **참고**: `requires_grad`의 값은 텐서를 생성할 때 설정하거나, 나중에 `x.requires_grad_(True)` 메소드로도 설정할 수 있다.

`nn.Linear` 같은 레이어를 만들면 내부의 `weight`, `bias`는 자동으로 `requires_grad=True`로 설정되어 있어서, 우리가 직접 켜지 않아도 이미 추적 준비가 되어 있다는 것도 참고할 만한 부분이었다.

---

## `Function` 객체와 `grad_fn`

"연산 그래프를 구성하기 위해 텐서에 적용하는 함수는 사실 `Function` 클래스의 객체입니다. 역방향 전파 함수에 대한 참조는 텐서의 `grad_fn` 속성에 저장됩니다"라는 문장의 의미를 자세하게 파고들어봤다.

텐서에 연산을 하면, 그 연산은 값만 계산하고 끝나는 게 아니라 **"이 연산을 어떻게 미분하면 되는지도 같이 알고 있는 특별한 객체(`Function`)"** 가 처리한다는 뜻이었다.

```python
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")
```

```
Gradient function for z = <AddBackward0 object at 0x7a37a0614580>
Gradient function for loss = <BinaryCrossEntropyWithLogitsBackward0 object at 0x7a37a06148e0>
```

각 텐서마다 "내가 어떤 연산으로 만들어졌는지"를 `grad_fn`에 기록해두고 있고, 이게 마치 이정표처럼 뒤에서부터 앞으로 거슬러 올라갈 수 있는 경로(연산 그래프)를 만들어준다. 그리고 `loss.backward()`를 호출하면 파이토치는 이 `grad_fn`들을 하나씩 따라가면서, 각 연산에 해당하는 미분 공식을 적용해 최종 기울기까지 계산해낸다.

---

## 변화도(Gradient) 계산하기

신경망에서 매개변수의 가중치를 최적화하려면 **매개변수에 대한 손실함수의 도함수(derivative)를 계산**해야 한다고 하는데, 왜 그런지 궁금했다.

도함수는 **"이 지점에서 입력을 살짝 바꾸면 결과가 어느 방향으로 얼마나 바뀌는가"** 를 알려주는 값이다. 이게 없으면 가중치를 어느 방향으로 바꿔야 손실이 줄어드는지 알 방법이 없고, 파라미터가 수백만 개인 신경망에서 무작정 시행착오로 찾는 건 현실적으로 불가능하다고 한다.

```python
loss.backward()
print(w.grad)
print(b.grad)
```

```
tensor([[0.2979, 0.0879, 0.2730], ...])
tensor([0.2979, 0.0879, 0.2730])
```

> **참고**
> - 연산 그래프의 잎(leaf) 노드 중 `requires_grad=True`인 노드들의 `grad` 속성만 구할 수 있다. 실제로 업데이트해야 할 대상은 가중치 같은 잎 노드들뿐이라서, 파이토치는 메모리를 아끼기 위해 중간 계산 결과(`z`, `loss` 등)의 변화도는 따로 저장하지 않는다.
> - 성능상의 이유로, 주어진 그래프에서 `backward`를 사용한 변화도 계산은 한 번만 수행할 수 있다. 동일한 그래프에서 여러 번 `backward`를 호출해야 하면 `retain_graph=True`를 전달해야 한다.

---

## 기울기를 보고 어떻게 판단하는가

#### 오해했던 점

기울기를 계산한다는 건 알겠는데, 그 값을 보고 구체적으로 어떻게 가중치를 조정하는지, 그리고 **언제가 "최적"** 인지 감이 안 왔다.

#### 실제로는

**기울기의 부호로 방향을, 크기로 얼마나 움직일지를 판단**한다.

```python
새로운 가중치 = 기존 가중치 - (학습률 × 기울기)
```

- 기울기가 양수 → 그 방향은 오르막 → 가중치를 줄여야 함
- 기울기가 음수 → 그 방향은 내리막 → 가중치를 늘려야 함

산을 내려가는 것에 비유하면, 기울기는 "지금 서 있는 지점에서 어느 방향이 오르막이고 내리막인지" 알려주는 나침반 역할이고, 항상 기울기의 반대 방향으로 이동해야 손실이 낮아지는 쪽으로 간다.

**"최적의 가중치" 는 그 지점의 기울기가 0** (더 이상 어느 쪽으로도 손실이 안 낮아지는 지점, 계곡의 가장 낮은 곳)이 되는 지점이다. 한 번에 그 지점을 찾는 게 아니라, 기울기 계산 → 반대 방향으로 조금 이동 → 다시 계산, 이 과정을 반복하며 서서히 접근한다.

---

## 변화도 추적 멈추기 — `torch.no_grad()`

#### 오해했던 점

`torch.no_grad()`를 쓰면, 그 전에 `requires_grad=True`로 켜뒀던 텐서도 갑자기 꺼지는 줄 알았다.

#### 실제로는

`torch.no_grad()`는 텐서의 `requires_grad` 값 자체를 바꾸는 게 아니라, **그 블록 안에서 새로 계산되는 결과의 추적만 잠깐 꺼두는 것**이었다.

```python
z = torch.matmul(x, w)+b
print(z.requires_grad)   # True

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)   # False
```

```python
w = torch.randn(3, requires_grad=True)
with torch.no_grad():
    y = w * 2   # 여기서는 추적 안 됨
z = w * 3   # 블록 밖으로 나오면 다시 정상적으로 추적됨
```

`w` 자체는 여전히 `requires_grad=True`로 남아있고, `torch.no_grad()` 블록을 벗어나면 다시 원래대로 추적이 켜진다. `detach()` 메소드로도 같은 효과(추적이 안 되는 새 텐서를 얻음)를 낼 수 있다.

```python
z_det = z.detach()
print(z_det.requires_grad)  # False
```

변화도 추적을 멈춰야 하는 이유는 두 가지다.
- 신경망의 일부 매개변수를 고정된 매개변수(frozen parameter)로 표시하고 싶을 때
- 학습이 끝난 뒤 순전파만 하면 될 때, 추적을 꺼서 연산 속도를 높이고 메모리를 아낄 수 있다.

---

## 방향성 비순환 그래프(DAG)

#### 오해했던 점

"방향성 비순환 그래프(DAG)"라는 용어 자체가 낯설었다.

#### 실제로는

이름을 그대로 풀면 이해가 됐다.

- **그래프(Graph)**: 노드(점)들이 엣지(선)로 연결된 구조
- **방향성(Directed)**: 그 연결선에 방향(화살표)이 있음
- **비순환(Acyclic)**: 화살표를 따라가도 절대 원래 출발점으로 돌아오지 않음

파이토치의 연산 그래프는 텐서(노드)들이 연산(엣지)으로 연결되어 있고, 화살표가 항상 입력 → 출력 방향으로만 흐르며, 절대 자기 자신으로 되돌아오는 순환 경로가 없다. DAG의 잎(leaf)은 입력 텐서고, 뿌리(root)는 결과 텐서다.

```
w (가중치) ──┐
              ├─→ [곱셈] ──→ y ──→ [덧셈] ──→ z ──→ [합계] ──→ loss
x (입력)   ──┘
```

순전파 단계에서 autograd는 두 가지를 동시에 한다: 연산을 수행해서 결과 텐서를 계산하는 것, 그리고 DAG에 연산의 변화도 기능(grad_fn)을 유지하는 것이다.

역전파는 DAG 뿌리에서 `.backward()`가 호출될 때 시작되며, 각 `grad_fn`으로부터 변화도를 계산하고, 각 텐서의 `.grad` 속성에 결과를 쌓아가며, 연쇄 법칙을 이용해 모든 잎 텐서까지 전파한다.

이 구조(순환이 없다는 것) 덕분에, `loss`에서 시작해서 `grad_fn`들을 거슬러 올라가면 반드시 유한한 단계 안에 최초 텐서(leaf node)까지 도달할 수 있다는 게 보장된다.

---

## DAG가 "동적(dynamic)"이라는 것

"PyTorch에서 DAG들은 동적입니다. 매번 `.backward()`가 호출되고 나면 새로운 그래프를 채우기 시작합니다"라는 문장과, 이게 왜 흐름 제어(`if`, `for`) 구문을 쓸 수 있게 해준다는 건지 연결이 안 됐다.

그래서 조사해보니, 파이토치의 계산 그래프는 미리 정해진 고정 도면이 아니라, **코드를 실행할 때마다 그때그때 새로 그려지는 것**이었다.

```python
for epoch in range(100):
    y = model(x)          # 이 순간, 계산 그래프가 즉석에서 만들어짐
    loss = loss_fn(y, target)
    loss.backward()        # 방금 만들어진 그래프를 따라 역전파
    # backward()가 끝나면 이 그래프는 사라짐
```

매 반복마다 그래프를 완전히 새로 그리기 때문에, 그 사이에 코드 흐름을 자유롭게 바꿀 수 있다. 예를 들어 모델의 `forward` 안에 `if`문이 있어서 실행마다 다른 레이어를 타더라도, 그래프는 실제로 실행된 경로 그대로 그려지기 때문에 문제가 없다.

```python
def forward(self, x):
    if x.sum() > 0:
        x = self.layer_a(x)
    else:
        x = self.layer_b(x)
    return x
```

그래프를 미리 고정해둬야 하는 정적 방식이었다면 이런 코드는 애초에 다루기 어려웠을 텐데, 동적 그래프는 실행된 경로 그대로 그때그때 그래프를 만들기 때문에 자연스럽게 처리된다는 것도 함께 이해하게 되었다.

---

## 야코비안 곱(Jacobian Product)

보통 지금까지 다룬 건 출력이 스칼라(숫자 하나, `loss`)인 경우였는데, **출력이 여러 개(벡터)일 때는 "입력 각각이 출력 각각에 얼마나 영향을 미쳤는지" 전부 알아야 해서 문제가 복잡해진다.** 이 모든 조합을 표(행렬)로 정리한 것이 야코비안 행렬이다.

```
야코비안 행렬 J =

∂y1/∂x1   ∂y1/∂x2   ∂y1/∂x3
∂y2/∂x1   ∂y2/∂x2   ∂y2/∂x3
∂y3/∂x1   ∂y3/∂x2   ∂y3/∂x3
```

신경망은 파라미터가 수백만 개라서 이 거대한 행렬 전체를 계산하고 저장하는 건 비효율적이다. 그래서 파이토치는 야코비안 행렬 전체 대신, **야코비안 행렬과 벡터 `v`를 곱한 결과(야코비안-벡터 곱)** 만 계산한다. 

`v^T · J`를 계산하는 방식이고, 이 과정은 `v`를 인자로 `backward`를 호출하면 이뤄진다. `v`의 크기는 곱을 계산하려는 원래 텐서의 크기와 같아야 한다.

```python
inp = torch.eye(4, 5, requires_grad=True)
out = (inp+1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")
```

동일한 인자로 `backward`를 두 번 호출하면 변화도 값이 달라진다는 점도 유의해야 한다. 파이토치는 역전파 시 변화도를 누적(accumulate)하기 때문에, 계산된 변화도 값이 그래프의 모든 잎 노드의 `grad` 속성에 계속 더해지기 때문이다.

그래서 제대로 된 변화도를 계산하려면 `grad` 속성을 먼저 0으로 만들어야 하며, 실제 학습 과정에서는 옵티마이저가 이 작업(`optimizer.zero_grad()`)을 대신 진행해준다.

> **참고**: 지금까지는 매개변수 없이 `backward()`를 호출했는데, 이는 본질적으로 `backward(torch.tensor(1.0))`을 호출하는 것과 같다. 손실처럼 스칼라 값 함수의 변화도를 계산할 때 유용한 방법중 하나이다.

---
*다음 편: 6편 Optimization에서 이어집니다.*