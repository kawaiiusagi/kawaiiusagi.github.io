---
title: "PyTorch 기초 학습 (6) — 모델 매개변수 최적화하기"
date: 2026-09-12
draft: false
tags: ["PyTorch", "딥러닝", "study-log"]
categories: ["PyTorch"]
---

> PyTorch 공식 튜토리얼을 학습하며 정리한 내용입니다.
> (https://tutorials.pytorch.kr/beginner/basics/optimization_tutorial.html)

---

## 하이퍼파라미터(Hyperparameter)

모델을 학습하는 과정은 반복적인 과정을 거친다. 각 반복 단계에서 모델은 출력을 추측하고, 추측과 정답 사이의 오류(손실)를 계산하고, 매개변수에 대한 오류의 도함수를 수집한 뒤, 경사하강법을 사용해 이 파라미터들을 최적화한다.

학습 시 정의하는 하이퍼파라미터는 다음과 같다.

- **에폭(epoch) 수** — 데이터셋을 반복하는 횟수
- **배치 크기(batch size)** — 매개변수가 갱신되기 전 신경망을 통해 전파된 데이터 샘플의 수
- **학습률(learning rate)** — 각 배치/에폭에서 모델의 매개변수를 조절하는 비율. 값이 작을수록 학습 속도가 느려지고, 값이 크면 학습 중 예측할 수 없는 동작이 발생할 수 있다.

```python
learning_rate = 1e-3
batch_size = 64
epochs = 5
```

#### 오해했던 점

처음에 "최적화 단계의 각 반복(iteration)을 에폭이라고 부른다"는 문장을 보고, 배치 하나를 처리할 때마다 에폭이 되는 줄 알았다.

#### 실제로는

에폭(Epoch)은 **전체 학습 데이터셋을 한 번 다 훑는 것**을 말하며, "이터레이션"과는 다른 개념이다.

```python
for epoch in range(10):              # 이 바깥 루프가 "에폭" (전체 데이터를 10번 반복)
    for batch in dataloader:          # 이 안쪽 루프 한 번이 "이터레이션(스텝)"
        ...
```

하나의 에폭 안에는 여러 번의 이터레이션이 들어있다.

```
전체 데이터: 60000개
배치 크기: 64개
1 에폭 = 60000 ÷ 64 ≈ 938번의 이터레이션(배치 처리)
```

---

## 손실 함수(Loss Function)

손실 함수는 예측 결과와 실제 값 사이의 틀린 정도를 측정하며, 학습 중에 이 값을 최소화하는 게 목표다. 

일반적인 손실 함수로는 회귀 문제에 쓰는 `nn.MSELoss`, 분류에 쓰는 `nn.NLLLoss`(음의 로그 우도), 그리고 `nn.LogSoftmax`와 `nn.NLLLoss`를 합친 `nn.CrossEntropyLoss` 등이 있다.

```python
loss_fn = nn.CrossEntropyLoss()
```

"음의 로그 우도(NLL)"와 `CrossEntropyLoss`가 각각 어떤 것인지 조사해보았다.

**우도(likelihood)** 는 모델이 정답에 부여한 확률이다.

```python
pred_probab = [0.05, 0.01, 0.35, 0.02, 0.01, 0.15, 0.03, 0.01, 0.06, 0.31]
정답 = 9
모델이 정답(9번)에 부여한 확률 = 0.31   # 이게 우도
```

**확률값을 그대로 여러 개 곱하면 계산이 불안정**해지기 때문에 로그를 씌운다. 다만 로그를 씌우면 1보다 작은 값은 음수가 되고, 모델이 정답을 잘 맞출수록 오히려 0에 가까운(더 작은) 음수가 되어 "손실은 작을수록 좋다"는 관례와 반대로 동작한다. 

그래서 앞에 마이너스를 붙여 부호를 뒤집는다.

```python
NLL = -log(정답에 대한 확률)

모델이 정답을 잘 맞춤 (확률 0.9) → -log(0.9) ≈ 0.10   (작은 값, 좋음)
모델이 정답을 잘 못 맞춤 (확률 0.1) → -log(0.1) ≈ 2.30   (큰 값, 나쁨)
```

**`CrossEntropyLoss`는 Softmax 변환과 NLL 계산을 한 번에 처리해주는 함수**다.

```python
# 원래는 두 단계를 따로 해야 함
pred_probab = nn.Softmax(dim=1)(logits)
loss = nn.NLLLoss()(torch.log(pred_probab), target)

# CrossEntropyLoss는 이 둘을 한 번에 처리
loss = nn.CrossEntropyLoss()(logits, target)
```

`CrossEntropyLoss`에 넣는 입력은 이미 Softmax를 거친 확률이 아니라 **원시 점수(logits) 그대로** 넣어야 한다. 내부에서 자동으로 Softmax를 적용해주기 때문이다.

**Sigmoid와 Softmax의 차이**도 짚어봤다. 둘 다 값을 0~1 사이로 압축하는 "정규화"를 하지만...
- Sigmoid는 숫자 하나만 독립적으로 보고 압축하는 반면(그래서 출력들을 다 더해도 1이 안 된다.)
- Softmax는 전체 숫자를 다 같이 고려해서(분모에 전체 합이 들어감) 압축하기 때문에 결과의 합이 항상 1이 된다.

```python
softmax(x_i) = e^(x_i) / Σ e^(x_j)   # 분모에 "전체 합"이 들어감
```

그래서 이진 분류(정답이 둘 중 하나)에는 Sigmoid, 다중 클래스 분류(여러 카테고리 중 하나)에는 Softmax를 쓴다.

---

## 옵티마이저(Optimizer)

최적화는 각 학습 단계에서 모델의 오류를 줄이기 위해 **모델 매개변수를 조정하는 과정**이다. 이 과정이 수행되는 방식을 정의하는 게 최적화 알고리즘이며, 모든 최적화 절차는 `optimizer` 객체에 캡슐화된다.

```python
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

학습 단계에서 최적화는 세 단계로 이뤄진다.

1. `optimizer.step()`으로 역전파 단계에서 수집된 변화도로 매개변수를 조정한다.
    {{< details summary="optimizer.step()이 매개변수를 조정하는 방식" >}}
`backward()`가 계산해둔 `.grad` 값을 이용해서, 각 가중치마다 다음 계산을 자동으로 실행한다.

```python
새로운 가중치 = 기존 가중치 - (학습률 × 기울기)
```

```python
# 예시
w        # tensor([0.5000, 0.3000, 0.8000])
w.grad   # tensor([0.2000, 0.1000, 0.4000])  ← backward()가 계산해둔 값

optimizer.step()

w  # tensor([0.4980, 0.2990, 0.7960])
#  0.5000 - (0.01 × 0.2000) = 0.4980
```

`optimizer = torch.optim.SGD(model.parameters(), lr=0.01)`에서 `model.parameters()`로 업데이트할 가중치들을 미리 등록해두기 때문에, `optimizer.step()`이 호출되면 등록된 모든 가중치를 하나씩 돌면서 이 계산을 적용한다. 

`Adam`, `RMSProp` 등 다른 옵티마이저는 이 계산 공식만 더 정교하게(관성, 적응형 학습률 등을 반영해서) 바꾼 것이다.
{{< /details >}}
2. `loss.backward()`로 예측 손실을 역전파한다. 파이토치는 각 매개변수에 대한 손실의 변화도를 저장한다.
3. `optimizer.zero_grad()`로 모델 매개변수의 변화도를 재설정한다. 기본적으로 변화도는 더해지기(add up) 때문에, 중복 계산을 막기 위해 반복마다 명시적으로 0으로 설정해야 한다.

본문에서는 `Adam`이나 `RMSProp`에 대해서 언급했는데, 해당 옵티마이저들이 어떤 성격을 가졌는지 궁금해 조사를 추가적으로 해보았다.

조사 결과, 이 기본 방식(SGD, 확률적 경사하강법)에 더해, `Adam`이나 `RMSProp` 같은 옵티마이저는 이동하는 방식을 더 똑똑하게 만든 것이었다.

- **SGD**: 항상 고정된 크기(학습률)로 이동
- **RMSProp**: 최근 기울기가 얼마나 요동쳤는지를 보고, 학습률을 상황에 따라 자동으로 조절 (요동이 심한 방향은 조심스럽게, 안정적인 방향은 크게 이동)
- **Adam**: RMSProp의 학습률 자동 조절 기능에, 이전에 가던 방향을 어느 정도 유지하려는 관성(momentum)까지 더한 방식

산길을 걷는 것에 비유하면, SGD는 항상 같은 보폭으로 걷는 사람, RMSProp은 길 상태(울퉁불퉁함)에 따라 보폭을 조절하는 사람, Adam은 거기에 더해 이전 방향으로 계속 나아가려는 관성까지 가진 사람이다. 

Adam이 대부분의 상황에서 안정적이고 빠르게 학습이 잘 되는 편이라 실무에서 기본 선택지로 가장 널리 쓰인다고 한다.

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

---

## 전체 구현 (train_loop / test_loop)

```python
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()   # 학습 모드로 설정
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    model.eval()   # 평가 모드로 설정
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

{{< details summary="참고사항" >}}
`model.train()`과 `model.eval()`은 배치 정규화(Batch Normalization) 및 드롭아웃(Dropout) 레이어가 있을 때 중요하다. 

지금 이 예시 모델에는 해당 레이어가 없어서 없어도 되지만, 모범 사례를 위해 넣어둔 것이다.{{< /details >}}

`test_loop` 안에서 `torch.no_grad()`로 감싸는 이유도, 이전 편(Autograd)에서 다뤘던 대로 테스트 시에는 역전파가 필요 없어서 변화도 계산 자체를 꺼서 불필요한 연산과 메모리 사용을 줄이기 위해서였다.

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
```

에폭이 반복될수록 손실(loss)이 점점 줄어들고 정확도(Accuracy)가 점점 올라가는 걸 로그로 직접 확인할 수 있었다.
```
Epoch 1: Accuracy: 46.4%, Avg loss: 2.145027
Epoch 5: Accuracy: 64.4%, Avg loss: 1.091447
Epoch 10: Accuracy: 71.0%, Avg loss: 0.788186
```

`optimizer.zero_grad()`를 매번 호출해야 하는 이유도 이전 편(Autograd)에서 확인했던 것과 이어지는 부분이었다.

파이토치는 역전파 시 변화도를 누적(accumulate)하기 때문에, 매 반복마다 초기화하지 않으면 이전 배치의 기울기가 계속 더해져서 잘못된 방향으로 학습될 수 있다는 점을 유의해야한다.

---
*다음 편: 7편 "모델 저장 및 불러오기"에서 이어집니다.*