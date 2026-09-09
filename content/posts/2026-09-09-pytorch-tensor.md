---
title: "PyTorch 기초 학습 (1) — Tensor"
date: 2026-09-09
draft: false
tags: ["PyTorch", "딥러닝", "study-log"]
categories: ["PyTorch"]
---

> PyTorch 공식 튜토리얼을 학습하며 정리한 내용입니다.
> (https://docs.pytorch.org/tutorials/index.html)

---

## 텐서란 무엇인가
- **딥러닝 관점** : GPU에서 병렬 연산이 가능하고 자동미분(autograd) 기능이 내장된 다차원 배열이다. 구조 자체는 넘파이 배열과 비슷하지만, 이 두 기능이 딥러닝 학습(backpropagation)을 가능하게 하는 핵심적인 차이점이다.

- **수학/물리학 관점** : 좌표계가 바뀔 때 정해진 규칙에 따라 값이 함께 변환되는 다중선형(multilinear) 객체. 스칼라(0차 텐서)는 좌표계를 바꿔도 값이 그대로 유지되고, 벡터(1차 텐서)는 좌표계를 회전시키면 그 변환 규칙에 맞춰 성분이 바뀐다.

---

## 텐서 생성 — `torch.tensor()`

`x_data = torch.tensor(data)`를 실행하면 내부적으로 몇 가지 일이 일어난다.

1. **dtype 자동 추론** — `[1,2,3]`이면 `int64`, `[1.0, 2.0]`이면 `float32`
2. **메모리 복사** — 항상 데이터를 복사해서 새 메모리에 텐서를 생성한다. (원본을 나중에 바꾸어도 영향이 없다.)
3. **디바이스 배정** — 기본값은 CPU
4. **`requires_grad`는 기본 `False`**

{{< details summary="참고: requires_grad가 True일 때" >}}
텐서에 들어간 모든 연산이 계산 그래프에 기록되고, 파이토치가 이 기록을 거꾸로 따라가면서 자동으로 미분 기울기를 계산해준다.
{{< /details >}}

```python
import numpy as np
data = np.array([1, 2, 3])
x = torch.tensor(data)
data[0] = 999
print(x)  # 여전히 [1, 2, 3] — 영향 없음
```

#### 오해했던 점

매번 데이터를 복사해서 새로운 텐서를 만드는 것보다, 그냥 형변환하듯 참조만 하는 게 더 효율적이지 않을까 생각했다.

#### 실제로는

파이토치는 목적에 따라 골라 쓸 수 있도록 세 가지 함수를 따로 제공한다.

| 함수 | 복사 여부 | 용도 |
|---|---|---|
| `torch.tensor(data)` | 항상 복사 | 안전하게 독립된 텐서가 필요할 때 (기본값) |
| `torch.as_tensor(data)` | 가능하면 안 함 | 성능이 중요할 때 |
| `torch.from_numpy(data)` | 절대 안 함 | 넘파이와 명시적으로 메모리 공유하고 싶을 때 |

`torch.tensor()`가 기본값으로 복사를 택한 이유는 두 가지다. 메모리를 공유하면 한쪽을 바꿨을 때 **다른 쪽도 의도치 않게 바뀌는 부작용**이 생길 수 있어 안전성을 우선한 것이고, 또한 애초에 파이썬 리스트는 **연속된 메모리 버퍼가 없어**서 공유 자체가 불가능하기 때문이다.

---

## `_like` 계열 함수 — 기존 텐서로부터 생성

#### 오해했던 점

"속성을 유지한다"는 말을 값까지 그대로 복사해온다는 뜻으로 오해했다.

#### 실제로는

`_like` 계열 함수는 기존 텐서의 shape과 dtype만 물려받고, 값은 함수 이름대로 새로 채운다. 틀(shape, dtype)만 가져오고 내용물(값)은 새로 채워지는 것이었다.

```python
x_data = torch.tensor([[1, 2], [3, 4]])
x_ones = torch.ones_like(x_data)   # shape/dtype 유지, 값은 전부 1
x_rand = torch.rand_like(x_data, dtype=torch.float)  # dtype 덮어씀, 값은 난수
```

`_like`에 넘기는 대상은 반드시 진짜 `torch.Tensor` 객체여야 한다. 리스트나 배열은 `.shape`/`.dtype`이 없거나 형식이 달라서 안 된다.

```python
torch.ones_like(x_data)
# 사실상: torch.ones(x_data.shape, dtype=x_data.dtype)
```

`_like`가 없는 버전(`torch.ones((2,3))` 등)은 참조할 기존 텐서 없이 shape을 직접 지정해서 만드는 것이다. shape은 튜플 변수로 미리 만들어서 넘길 수도 있다.

```python
shape = (2, 3)
torch.rand(shape)     # 튜플로 전달
torch.rand(*shape)    # 언패킹해서 전달
torch.rand(2, 3)      # 직접 나열 — 결과 다 동일
```

| 상황 | 방법 |
|---|---|
| shape을 직접 알고 지정 | `torch.ones((2,3))` |
| 기존 텐서와 같은 shape 필요 | `torch.ones_like(x)` |
| 리스트/배열 → 텐서 | `torch.tensor(data)` |

---

## GPU / CUDA

#### 오해했던 점

GPU를 한 번 켜두면 계속 유지되는 전역 스위치 같은 걸 상상했다.

#### 실제로는

`.to("cuda")`는 그 텐서 변수 하나에만 적용된다. 텐서마다 개별적으로 위치를 지정하는 방식이며, 일부 연산은 CPU가 더 빠르거나 특정 텐서만 GPU 메모리에 올려야 하는 상황을 고려해서 파이토치가 텐서 단위로 세밀하게 제어하게 만든 것으로 보인다.

또한 `torch.cuda.is_available()`은 GPU 사용 가능 여부를 **확인만** 하는 함수이지, 그 자체로 GPU를 구동하지는 않는다.

```python
tensor = torch.rand(2, 3)
if torch.cuda.is_available():
    tensor = tensor.to("cuda")     # 이 tensor만 GPU로 이동

new_tensor = torch.rand(2, 3)      # 여전히 CPU
```

*(참고: `torch.set_default_device('cuda')`로 기본 device 자체를 바꾸는 방법도 있지만 흔히 쓰이지는 않는다.)*

---

## 인덱싱 / 슬라이싱

```python
tensor = torch.arange(16).reshape(4, 4)
tensor[0]     # [0,1,2,3] ← 행
tensor[:, 0]  # [0,4,8,12] ← 열
```

- `tensor[i]` → i번째 **행**
- `tensor[:, j]` → j번째 **열**
- `tensor[:, j] = 값` → j번째 열 전체를 덮어씀

---

## 텐서 이어붙이기 — `cat` / `stack`

#### 오해했던 점

`torch.cat`은 결국 "이어붙이기"인데, 1차원 벡터만 다룰 땐 방향이 하나뿐이라 `dim`이라는 파라미터가 왜 필요한지 잘 와닿지 않았다.

#### 실제로는

2차원 이상으로 가면 이야기가 달라진다. 같은 두 텐서를 세로로 붙일지 가로로 붙일지에 따라 결과 shape 자체가 완전히 달라지기 때문에, `dim`으로 방향을 명시하지 않으면 애초에 연산이 불가능하다.

```python
tensor = torch.tensor([[1,2,3],[4,5,6]])
torch.cat([tensor, tensor], dim=0)  # 세로로 붙음 (행 증가) → (4,3)
torch.cat([tensor, tensor], dim=1)  # 가로로 붙음 (열 증가) → (2,6)
```

`axis`(넘파이)와 `dim`(파이토치)은 완전히 같은 역할이며 이름만 다르다.
- `dim=0` → 세로(아래로), 행 증가
- `dim=1` → 가로(오른쪽으로), 열 증가

참고로 `torch.stack`은 `cat`과 달리 새로운 차원을 추가하며 합친다.

---

## 행렬곱 & 원소별 연산

#### 오해했던 점

`matmul()`과 `@`가 다른 연산인 줄 알았고, `out` 파라미터가 왜 필요한지도 감이 안 왔다.

#### 실제로는

`@`는 `matmul()`의 문법적 설탕(syntactic sugar)일 뿐 완전히 동일한 연산이다.

```python
tensor @ tensor.T
tensor.matmul(tensor.T)
torch.matmul(tensor, tensor.T)
# 셋 다 동일
```

주의할 점은 `*`(원소별 곱셈)와는 완전히 다른 연산이라는 것이다.

`out` 파라미터는 연산 결과를 새 메모리에 만드는 대신 미리 지정한 텐서 메모리에 덮어써서 저장하는 용도였다. 반복 연산이 많은 경우 메모리 할당 오버헤드를 줄일 수 있다.

```python
torch.matmul(tensor, tensor.T, out=y3)
```

요소별 연산에서도 브로드캐스팅이 적용된다. 넘파이와 동일한 규칙으로, shape이 완전히 같지 않아도 조건만 맞으면 자동으로 크기를 맞춰 계산한다.

```python
a = torch.tensor([[1,2,3],[4,5,6]])  # (2,3)
b = torch.tensor([10,20,30])          # (3,)
a * b  # [[10,40,90],[40,100,180]]
```

---

## NumPy와의 관계

#### 오해했던 점

"메모리 공간을 공유한다"는 문장을 보고, 그냥 값이 비슷하게 유지되는 정도로 이해했다.

#### 실제로는

값을 복사하지 않고 같은 메모리 주소를 그대로 같이 쓴다는 훨씬 강한 의미였다. 그래서 CPU 상의 텐서와 NumPy 배열 중 하나를 변경하면 다른 하나도 함께 변경된다.

```python
t = torch.ones(5)
n = t.numpy()
t.add_(1)
print(n)   # [2. 2. 2. 2. 2.] ← 같이 바뀜
```

구조가 비슷한 CPU 메모리 안에서 복사 없이 공유하면 변환이 빠르고 메모리도 절약된다. 독립적인 복사본이 필요하면 `.clone()`을 사용한다. GPU 텐서는 예외로, 넘파이와 공유가 불가능해 `.cpu()`로 옮긴 뒤 변환하며 이때는 복사가 일어난다.

---

## 자주 쓰는 함수들

#### `torch.randint`와 `.item()`의 역할.

- `torch.randint(low, high, size)` — 지정 범위 정수 난수를 **텐서**로 생성
- `.item()` — 원소 1개짜리 텐서를 **순수 파이썬 숫자**로 변환 ("텐서라는 포장지를 벗기고 안에 있는 순수한 숫자 값만 남기는 것")

```python
idx = torch.randint(0, len(training_data), (1,))
idx.item()  # 텐서 → 파이썬 int
```

전체 흐름을 코드 한 줄로 보면:

```python
sample_idx = torch.randint(len(training_data), size=(1,)).item()
```

1. `len(training_data)` → 전체 개수 (예: 60000)
2. `torch.randint(60000, size=(1,))` → low 생략 시 기본값 0, "0~60000 미만 정수 하나를 텐서로" → `tensor([37482])`
3. `.item()` → 텐서 껍데기를 벗기고 파이썬 정수로 → `37482`

#### 오해했던 점

`.squeeze()`가 "1인 차원을 없앤다"는 게 정확히 어떤 조건에서 동작하는지 헷갈렸다.

#### 실제로는

크기가 1인 차원만 골라서 제거하고, 1이 아닌 차원을 지정하면 에러 없이 그냥 무시된다.

```python
x = torch.ones(1, 3, 1, 4)
x.squeeze()    # (3, 4) — 크기 1인 차원 전부 제거
x.squeeze(1)   # 1번째 차원(크기 3, 1 아님) → 아무 변화 없음
x.squeeze(0)   # 0번째 차원(크기 1) → 실제 제거됨
```

---

## 시각화 (matplotlib)

#### 오해했던 점

`cmap="gray"`를 빼먹고 그렸더니 흑백이어야 할 이미지가 이상한 보라-노랑 색으로 나왔다. matplotlib이 "이건 픽셀값이니까 알아서 흑백 처리하겠지"라고 판단하는 줄 알았다.

#### 실제로는

그런 판단 로직이 있는 게 아니라, `imshow()`는 항상 숫자를 색으로 변환하고 지정을 안 하면 기본 컬러맵(`viridis`, 낮은 값=보라·높은 값=노랑)이 적용될 뿐이었다.

```python
plt.imshow(img.squeeze(), cmap="gray")
plt.axis("off")
```

**<추가>**
- `img.squeeze()` — `(1,28,28)` → `(28,28)`로 채널 차원 제거
- `cmap="gray"` — 컬러맵을 흑백으로 지정, 낮은 값=검정·높은 값=흰색으로 원래 흑백처럼 보이게 함
- `plt.axis("off")` — 좌표 눈금/테두리 숨김

---

*다음 편: 2편 Dataset & DataLoader에서 이어집니다.*
