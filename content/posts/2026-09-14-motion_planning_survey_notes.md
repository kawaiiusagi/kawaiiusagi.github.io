---
title: "A Survey of Motion Planning and Control Techniques for Self-Driving Urban Vehicles - (2)"
date: 2026-09-14
draft: false
tags: ["Paper_Review", "Autonomous Driving", "study-log"]
categories: ["Survey_Notes"]
---
| 원문 링크 : https://arxiv.org/pdf/1604.07446
- **제목** : A Survey of Motion Planning and Control Techniques for Self-Driving Urban Vehicles
- **저자**: B. Paden, M. Čáp, S. Z. Yong, D. Yershov, E. Frazzoli
- **분류**: Survey (자율주행 Planning & Control)
- **읽은 범위**: Section IV-A (Path Planning), IV-B (Trajectory Planning)

## 한 줄 요약

> Motion Planning의 핵심 두 문제(Path Planning, Trajectory Planning)를 수학적으로 formal하게 정의하고, **두 문제 모두 계산적으로 극도로 어렵다는 것(PSPACE-hard, NP-hard)**을 보인 뒤, 그래서 정확한 해 대신 근사적 수치해법 3가지 계열이 쓰인다는 것을 소개하는 파트.

## 핵심 개념 정리

### Configuration Space (구성 공간)

차의 상태를 표현하는 데 필요한 최소 정보의 집합. 기본 모델에서는 **(x, y, θ)** 세 가지 — 위치(x, y)와 방향(θ, heading angle) — 으로 충분하다.

- Configuration space **X**: (x, y, θ)가 가질 수 있는 모든 조합의 (추상적) 공간
- **X_free**: 장애물과 충돌하지 않는 등, 차가 실제로 있어도 되는 configuration들만 모은 부분집합
- **X_goal**: 목표 구역
- **x_init**: 출발 시점의 configuration

차가 이동한다는 것은 결국 이 configuration space 안에서 점이 이동하는 것이고, 그 궤적을 찾는 게 motion planning의 본질이라는 걸 이해하고 나니 이후 수식들이 훨씬 잘 읽혔다.

### Holonomic vs Non-holonomic Constraint

- **Holonomic constraint**: 위치(configuration) 자체에 대한 제약. "이 자리에는 있으면 안 된다" (장애물, 도로 경계 등)
- **Non-holonomic constraint**: 움직이는 방식에 대한 제약. 자동차가 옆으로 미끄러지듯 이동하지 못하고, 조향각에 따라서만 방향을 바꿀 수 있는 것을 대표적인 예시로 말할 수 있다.

### Path vs Trajectory — "prescribe"의 의미 차이

- **Path σ(α)**: α ∈ [0,1]은 시간이 아니라 **경로 진행률**. "어떤 순서로 지나가는지"만 정의하고, **언제** 그 지점에 도달하는지는 정하지 않는다 → "does not prescribe" 표현이 나온 이유.
- **Trajectory π(t)**: t ∈ [0,T]는 실제 **시간**. "몇 초 후에 어디에 있어야 하는지"까지 다 정해준다 → "prescribes" 표현이 나온 이유.

이 차이 때문에 동적 장애물(다른 차, 보행자 등)과의 **충돌을 판단하려면 반드시 trajectory(시간 정보 포함)가 필요**하다는 것도 자연스럽게 이해가 됐다. Path만으로는 "저 사람이 3초 후 어디 있을지"와 내 경로를 시간 축에서 매칭할 방법이 없기 때문이다.

## 계산 복잡도 — 왜 근사적 방법을 쓸 수밖에 없는가

### IV-A: Path Planning

- 문제를 Problem IV.1로 formal하게 정의한 후, 이 문제가 **PSPACE-hard**임을 보임
- PSPACE-hard는 NP-hard보다도 어려운 축에 속하는 문제로, **P ≠ NP라는 (거의 정설로 받아들여지는) 가정 하에서는 다항시간(polynomial-time) 안에 정확한 해를 구하는 알고리즘이 존재하지 않는다**는 뜻
- 다만 조건에 따라 난이도가 달라진다는 "조건표" 형태로 여러 결과가 나열됨:
  - 장애물 없는 홀로노믹 로봇 → 다항시간에 풀림
  - 폴리곤 장애물 있는 홀로노믹 로봇 → PSPACE-complete
  - 정적 장애물 사이 최단경로(홀로노믹) → 다항시간 가능
  - 곡률 제한(자동차처럼) 붙은 경로 → NP-hard

→ 그래서 정확한 해 대신 **Variational methods(비선형 최적화), Graph-search methods(공간을 그래프로 이산화 후 최소비용 탐색), Incremental search(RRT처럼 점진적으로 트리를 확장)** 세 계열의 수치적 근사 방법이 쓰인다.

### IV-B: Trajectory Planning

- IV-A보다 짧고, 새로운 방법론을 소개하기보다는 **"왜 Path planning보다 더 어려운가"만 증명하는 파트**
- 핵심 근거: 정적 환경에서는 다항시간에 풀리던 문제가, **똑같은 문제를 동적 환경(움직이는 장애물) 버전으로 바꾸면 갑자기 풀기 어려워진다(intractable)**
  - 예: 정적 2D 폴리곤 환경에서 홀로노믹 점 로봇의 최단경로 → 다항시간
  - 같은 문제를 움직이는 장애물 버전으로 바꾸면 → NP-hard (Canny & Reif)
- 결론은 IV-A와 동일: 정확한 해 대신 numerical method를 쓸 수밖에 없다는 것

## 헷갈렸던 점 → 정리

**Q. "Moreover, trajectory planning ... harder than path planning" 문장이 무슨 말인지?**

→ "동적 환경을 고려하는 순간, 정적 환경에서는 잘 풀리던 문제가 갑자기 어려워진다"는 의미였다. 즉 문제의 본질이 바뀌는 게 아니라, **환경이 정적이냐 동적이냐라는 조건 하나가 계산 복잡도를 완전히 바꿔놓는다**는 것이 핵심이었다.

**Q. 수식이 낀 문장을 읽을 때 자꾸 세부 연결고리가 안 잡혔던 이유?**

→ 1회독 때는 수식 기호(x_init, X_free 등)를 대충 넘기면서 흐름만 파악했는데, 2회독에서 정확히 이해하려다 보니 그 기호들에 계속 걸려서 문장 흐름이 끊겼던 것 같다. 기호 정의(x_init=시작점, X_free=갈 수 있는 공간, X_goal=목표구역, D=부드러움 제약)를 먼저 정리하고 다시 읽으니 훨씬 매끄럽게 읽혔다.

**Q. P / NP / PSPACE가 어떤 것을 의미하나?**

- **P**: 컴퓨터가 빠르게(효율적으로) 풀 수 있는 문제들의 집합
- **NP**: 답이 맞는지 "확인"은 빠르게 할 수 있는 문제들 (푸는 것 자체는 오래 걸릴 수 있음)
- **PSPACE**: NP보다 더 넓은 범위. 시간은 오래 걸려도 되지만 메모리(공간)만 적당히 쓰면 풀 수 있는 문제들의 집합 (P ⊆ NP ⊆ PSPACE)

> 즉 이 논문에서 path/trajectory planning이 **PSPACE-hard**라고 한 것은, NP-hard보다도 더 어려운 축에 속하는 문제라는 뜻이고, 그만큼 정확한 해를 다항시간에 구하는 알고리즘이 존재할 가능성이 이론적으로 거의 없다는 걸 보여주는 근거였다.

- 위 개념 자체는 이번에 처음 접했지만, 장애물 유무·정적/동적 여부·곡률 제약 유무에 따라 같은 문제도 난이도가 P → NP-hard → PSPACE-hard로 바뀌는 구체적 사례들을 보니 추상적인 이론이 실제 문제에 어떻게 적용되는지 감이 잡혔다.

## 총평

이번 파트를 읽으면서 느낀 건, Motion Planning이라는 분야가 생각보다 **순수 수학/이론 컴퓨터과학(계산복잡도 이론)과 굉장히 맞닿아 있다**는 점이다. 단순히 "충돌 안 나게 경로 그리기"라는 직관적인 문제인 줄 알았는데, 그 이면에는 PSPACE-hardness 증명, P vs NP 같은 이론적 배경이 촘촘하게 깔려 있었고, 왜 정확한 해가 아니라 근사적 방법을 쓸 수밖에 없는지에 대한 논리가 상당히 엄밀하고 체계적으로 정립되어 있다는 걸 알게 됐다.

이 분야에서 연구를 하려면 로보틱스/제어 지식뿐 아니라 **알고리즘의 계산 복잡도를 다루는 이론적 기반**도 상당히 탄탄해야겠다는 생각이 들었다. 특히 "이 문제가 왜 어려운가"를 증명하는 논리 자체가 연구의 중요한 축이라는 걸 이번에 처음 체감했고, 이런 엄밀한 문제 정의와 증명 스타일에 익숙해지는 것 자체가 앞으로 이 분야를 계속 공부하는 데 필요한 능력이라는 생각이 들었다.

다음 서베이(다른 하위분야)를 읽을 때는, 오늘처럼 수식이 나오면 바로 뜻을 파악하려 하기보다 먼저 기호부터 정리하고 들어가는 방식을 적용해봐야겠다.

## 다음에 더 볼 것

- IV-C~E (Variational / Graph-search / Incremental search의 구체적 알고리즘) — 이번 회독에서는 스킵, 필요시 재방문
- 다른 하위분야(Perception 또는 학습 기반 자율주행) 서베이 1편
- 오늘 다룬 개념(configuration space, holonomic constraint) 기반의 간단한 파이토치/파이썬 프로젝트 아이디어 구체화
