---
title: "A Survey of Motion Planning and Control Techniques for Self-Driving Urban Vehicles - (1)"
date: 2026-09-13
draft: false
tags: ["Paper_Review", "Autonomous Driving", "study-log"]
categories: ["Survey_Notes"]
---
| 원문 링크 : https://arxiv.org/pdf/1604.07446
- **제목** : A Survey of Motion Planning and Control Techniques for Self-Driving Urban Vehicles
- **저자**: B. Paden, M. Čáp, S. Z. Yong, D. Yershov, E. Frazzoli
- **분류**: Survey (자율주행 Planning & Control)
- **읽은 범위**: Introduction, Section II (전체)

## 한 줄 요약

자율주행 차량의 의사결정을 상위(경로)에서 하위(핸들 조작)까지 4단계 계층으로 나누고, 각 단계에서 쓰이는 알고리즘과 수학적 접근법을 체계적으로 정리한 서베이

## 핵심 구조: 4단계 Decision-Masking Hierarchy

{{< mermaid >}}
flowchart TD
    A["Route Planning(어느 도로로?)"] --> B["Behavioral Decision Making(지금 뭘 할까?)"]
    B --> C["Motion Planning(어떤 구체적 경로로?)"]
    C --> D["Vehicle Control(어떻게 실제로 따라갈까?)"]

    A -.문제 성격.-> A1["그래프 탐색(조합 문제)"]
    B -.문제 성격.-> B1["이산적 선택(논리 문제)"]
    C -.문제 성격.-> C1["궤적 계산(최적화/기하 문제)"]
    D -.문제 성격.-> D1["피드백 제어(에러 최소화)"]
{{< /mermaid >}}

| 단계 | 질문 | 추상화 수준 | 대표 기법 |
|---|---|---|---|
| A. Route Planning | 어느 도로로 갈까 | High (거시적) | Dijkstra, A* |
| B. Behavioral Decision Making | 지금 어떤 행동을 할까 (차선변경/정지/양보) | 중상 | 유한상태기계, 강화학습 |
| C. Motion Planning | 그 행동을 어떤 구체적 경로/궤적으로 실행할까 | 중하 | RRT, 최적화 기반 궤적 생성 |
| D. Vehicle Control | 그 경로를 물리적으로 어떻게 따라갈까 | Low (물리적 실행) | Pure Pursuit, MPC, Lyapunov 기반 제어 |

## 왜 이렇게 4단계로 나눴을까? (스스로 답해본 것)

**Q. Behavioral Decision과 Motion Planning을 굳이 왜 분리했을까? 하나로 합치면 안 되나?**

→ 두 문제의 수학적 성격이 근본적으로 다르기 때문으로 보인다. 

Behavioral Decision은 "차선변경 하냐 마냐" 같은 **이산적(discrete) 선택**의 문제라 조합/논리적 접근(상태기계, 결정트리)이 자연스러운 반면, Motion Planning은 "어떤 곡선으로 이동하냐"는 **연속적(continuous) 값**을 다루는 최적화/기하 문제라 접근법 자체가 다르다. 

문제 성격이 다르면 알고리즘 설계도 완전히 달라지기 때문에 계층을 분리한 것으로 추정된다.

**Q. Low-level / High-level의 의미는?**

처음엔 "소프트웨어 관점의 저수준(코드/드라이버 레벨)"으로 착각했다.

하지만 실제로는 **물리적 실행에 얼마나 가까운가**를 뜻하는 표현이라고 한다. 


Route Planning(도로 그래프 위의 추상적 경로)이 high-level이고, Vehicle Control(실제 조향각/가속도 명령)이 low-level. 로보틱스·제어공학 전반에서 공통적으로 쓰이는 관례로 보인다.

## 헷갈렸던 점 → 정리

**Q. Behavioral Decision Making 단계에서 보행자 같은 장애물도 "인지"하는 걸 배우는 줄 알았는데?**

→ 아니었다. 장애물을 감지/인식하는 건 **Perception(인지)** 영역의 역할이고, Behavioral Decision Making은 "이미 인지된 정보(저기 보행자가 있다)"를 전제로 그 다음 행동(멈출지 돌아갈지)을 결정하는 단계이다. 즉 "인지를 학습"이 아니라 "행동 결정을 학습"이 정확한 표현이라고 할 수 있다.

**Q. Vehicle Control은 "차체 내부 모니터링"인 줄 알았는데?**

→ 정확히는 **피드백 제어 루프**이다. (1) 목표 경로/속도와 (2) 센서로 측정한 실제 상태 사이의 **에러를 계산**하고, (3) 그 에러를 줄이는 방향으로 조향/가속 명령을 계속 갱신하는 과정. 지속 감시라기보다, 매 순간 "지금 상태에서 최적 보정값은 뭘까"를 반복 계산하는 것을 말한다.

## 기존 지식과의 연결

- Route Planning의 Dijkstra는 자료구조/알고리즘 수업에서 배운 최단경로 알고리즘과 동일한 원리이다. 다만 **실제 도로 네트워크는 노드 수가 훨씬 많아** A* 같은 휴리스틱 탐색이 실무에서 더 자주 쓰이는 이유를 이해할 수 있었다.
- Vehicle Control의 피드백 루프 개념은 제어공학 기초(목표값-측정값-에러-보정)와 동일한 구조이다. MPC는 여기에 "미래 몇 스텝을 예측해서 미리 최적화"하는 게 추가된 확장판으로 이해된다.

## 비판적으로 본 점

이 4단계 파이프라인 구조는 전통적(규칙 기반) 자율주행 시스템 설계를 잘 설명하지만, 최근 End-to-end 학습 방식은 이 계층 구분 자체를 없애고 센서 입력에서 제어 명령까지 하나의 신경망으로 통째로 학습하는 방향으로 가고 있다고 한다. 

즉 이 논문은 "고전적/모듈화된 접근"을 다루는 것이고, 최신 트렌드(End-to-end, LLM 기반 추론 등)와는 다른 패러다임이라는 점을 염두에 두고 읽을 필요가 있을 것 같다.

## 다음에 더 볼 것

- IV-D Graph Search Methods (RRT, A* 실제 동작 방식 상세)
- 최근 강화학습 기반 Motion Planning 서베이 (2025년 최신 트렌드 확인용)
- End-to-end 방식과의 비교 (파이프라인 vs 통합 학습)
