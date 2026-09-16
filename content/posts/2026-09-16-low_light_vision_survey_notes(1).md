---
title: "Deep Learning for Low-Light Vision: A Comprehensive Survey - (1)"
date: 2026-09-16
draft: false
tags: ["Paper_Review", "Autonomous Driving", "Low-Light Vision", "study-log"]
categories: ["Survey_Notes"]
---
| 원문 링크 : https://ieeexplore.ieee.org/document/11018619
- **제목**: Deep Learning for Low-Light Vision: A Comprehensive Survey
- **저자**: Qian Zhao, Gang Li, Bin He, Runjie Shen (Tongji University)
- **학술지**: IEEE Transactions on Neural Networks and Learning Systems (TNNLS), Vol. 36, No. 9, 2025
- **분류**: Survey (저조도 비전 전반 — 화질 개선 + 고수준 인식 태스크)
- **읽은 범위**: Abstract, Introduction, Section III (Object Detection, A~F, Fig.5), Section VIII (Challenges & Future Directions) — 정독 / Table III, Section IV-A, VII-B, VII-C, Section V — 요약 정리 후 재방문 예정

## 한 줄 요약

> 저조도 환경에서의 비전 문제를 **화질 중심(Visual quality-driven)** 과 **인식 중심(Recognition quality-driven)** 두 축으로 나눠, LLIE(저조도 이미지 향상)와 저조도 객체 탐지 방법론을 종합적으로 정리한 서베이. 기존 서베이들이 이 둘을 따로 다뤘던 것과 달리, 두 영역의 연결고리(target inconsistency 등)를 짚어낸 게 차별점이다.

## 핵심 개념 정리

### 두 축의 분류 체계 (Fig. 2, Taxonomy)

{{< mermaid >}}
flowchart TD
    A["Low-Light Vision(저조도 비전 문제)"] --> B["Visual Quality-driven(화질을 좋게)"]
    A --> C["Recognition Quality-driven(인식을 잘하게)"]

    B --> B1["Low-Light Image Enhancement"]
    B1 -.학습 방식.-> B1a["Supervised / Unsupervised / Semi-supervised / Zero-shot"]
    B1 -.문제 성격.-> B1b["이미지 복원(사람 눈 기준 최적화)"]

    C --> C1["Low-Light Object Detection"]
    C --> C2["Other High-Level Tasks(Segmentation, Tracking 등)"]

    C1 -.전달 경로.-> C1a["Preprocessing / Feature Fusion / Image Darkening / Domain Adaptation / Multimodal"]
    C1 -.문제 성격.-> C1b["위치+분류 판단(기계 인식 기준 최적화)"]

    B1b -.공통 난제.-> X["Target Inconsistency(화질↔인식 목표 불일치)"]
    C1b -.공통 난제.-> X
{{< /mermaid >}}

### ψ⁻¹과 ζ — 저조도 객체 탐지의 공통 수식 틀 (Section III-A)

```
(f₁, f₂, ..., fₙ, f_L) = ψ⁻¹(I_L, ω)   ... 저조도 이미지에서 특징 추출/복원
(B, C) = ζ(f₁, f₂, ..., fₙ, f_L)       ... 그 특징으로 바운딩박스(B)+클래스(C) 판단
```

- **ψ⁻¹**: 저조도 이미지에서 어두워서 손실된 정보를 복원/추출하는 알고리즘이다. (실제 구현은 카테고리마다 다르다고 한다 --> 완성 이미지를 만들 수도, 중간 특징만 뽑을 수도 있다.)
- **ζ**: 그 결과를 받아 물체 위치(B)와 종류(C)를 판단하는 탐지기이다.
- 뒤에 나오는 5개 카테고리(B~F)는 결국 **이 두 단계를 어떻게 연결하느냐의 차이**이다.

### 저조도 객체 탐지 5개 카테고리 비교 (Section III-B~F)

| 카테고리 | 접근 방식 | 핵심 키워드 |
|---|---|---|
| B. Preprocessing | 이미지를 먼저 완성 → 탐지 | 픽셀 공간, 순차적 처리 |
| C. Feature Fusion | 완성 이미지 없이 특징끼리 바로 융합 | 특징 공간, 동시적 처리 |
| D. Image Darkening | 정상광 데이터를 인공적으로 어둡게 만들어 학습 데이터 확보 | 데이터 증강 |
| E. Domain Adaptation | 정상광에서 배운 지식을 저조도 도메인에 이식 | 도메인 격차 해소 |
| F. Multimodal | 적외선·LiDAR 등 다른 센서 정보를 추가 융합 | 센서 융합 |

- **B (Preprocessing)**: 초기엔 향상과 탐지가 완전히 분리돼 있었으나(예: 밝기조절 곡선 + 사전학습된 YOLOv3), 최근엔 **end-to-end**로 묶어서 탐지 손실(detection loss)이 향상 단계에도 반영되게 함 — "예쁘게 밝히기"가 아니라 "탐지가 잘 되게 밝히기"로 목표가 진화
- **C (Feature Fusion)**: 픽셀 공간에서 완성 이미지를 만드는 대신 특징 공간에서 바로 작업 → 효율성↑, 향상 과정의 왜곡이 탐지 단계까지 전파되는 문제↓.
{{< details summary="예시" >}}
- 매 프레임 화면 전체를 분석하는 대신, **각 프레임에서 1픽셀 두께의 가로선 하나만 샘플링한다.** 
- 이 선들을 시간 순서대로 이어붙이면 도로를 위에서 훑은 듯한 하나의 이미지(road profile image)가 만들어지고, 이를 기반으로 Segmentation을 수행한다. 
- 수행하므로써, **분석할 데이터양 자체가 줄어들어** 차량 주행 속도를 따라갈 수 있는 실시간 처리가 가능해진다.
{{< /details >}}
- **D (Image Darkening)**: 저조도 라벨 데이터 부족 문제(VIII절 챌린지 1번)를 정면으로 해결하려는 접근. COCO/VOC/KITTI 같은 기존 정상광 데이터셋을 인위적으로 어둡게 만들어 라벨을 그대로 물려받음
- **E (Domain Adaptation)**: Domain Adaptation의 세 전략 
  - **Domain-invariant feature learning (GRL)**: 도메인 판별기를 속이도록 학습시켜, 저조도든 정상광이든 공통으로 통하는 특징을 뽑아냄
  - **Domain translation (CycleGAN)**: 이미지 스타일 자체를 정상광↔저조도로 변환시켜 격차를 줄임
  - **Self-training (pseudo-label)**: 신뢰도 높은 예측을 가짜 정답 삼아 반복 학습시킴
- **F (Multimodal)**: RGB 단독의 한계(정보량 부족)를 적외선·LiDAR로 보완하지만 센서 비용·정렬(alignment) 문제가 트레이드오프

### Fig. 5 — 5개 카테고리를 관통하는 시각화
{{< mermaid >}}
flowchart TD
    P["문제 상황: L(저조도, 라벨 없음) vs H(정상광, 라벨 있음)"]

    P --> B["B. Preprocessing<br/>L → E(L) → O(E(L))"]
    P --> C["C. Feature Fusion<br/>L·H의 특징이 중간에서 융합"]
    P --> D["D. Image Darkening<br/>H → D(H) (역방향, 라벨 물려받음)"]
    P --> E["E. Domain Adaptation<br/>L ↔ H (양방향 격차 해소)"]
    P --> F["F. Multimodal<br/>L·H + 다른 센서 데이터 융합"]

    B -.방향.-> B1["순차적(low→high)"]
    C -.방향.-> C1["동시적(수렴/교차)"]
    D -.방향.-> D1["역방향(high→low)"]
    E -.방향.-> E1["쌍방향"]
    F -.특징.-> F1["카메라 외 정보원 추가"]
{{< /mermaid >}}

→ 결국 이 6개 패널은 "저조도엔 라벨이 없다"는 하나의 문제를 **다섯 가지 완전히 다른 철학**으로 풀고 있다는 걸 보여주는 그림.

## 헷갈렸던 점들

**Q. Feature Fusion(C절)이 Preprocessing(B절)과 뭐가 다른지 이해가 안 갔다.**

→ B절은 "향상을 완전히 끝낸 이미지 한 장"을 만들고 그걸 탐지기에 통째로 넣는 **순차적** 구조인 반면, C절은 완성 이미지를 만드는 과정 자체를 생략하고 **중간 특징(feature)** 단계에서 두 네트워크가 정보를 바로 섞어버리는 구조라는 게 핵심 차이였다.

## VIII절 — Challenges & Future Directions 요약

### 기존 4가지 난제
1. **데이터 부족(Lack of Data With Annotations)**: 지도학습(SL) 패러다임이 지배적인데 정작 라벨 있는 저조도 데이터가 부족함
2. **Target Inconsistency**: 화질 개선(사람 눈 기준)과 인식 정확도(기계 기준)의 목표가 달라서 통합 프레임워크 없이 각자 발전 → 시너지 부족
3. **Unknown Noises and Artifacts**: shot/read/thermal noise 등 종류가 다양하고 센서·환경마다 양상이 달라 일반화가 어려움
4. **Nonuniform Illumination**: 한 장면 안에서도 밝은 곳과 어두운 곳이 공존해 언더/오버 노출을 동시에 다뤄야 하는 어려움

### 6가지 미래 방향
1. **Real Datasets**: 더 다양한 환경·조도의 대규모 실제 데이터셋 확보
2. **Unified Model**: 여러 태스크·열화 패턴을 하나의 통합 모델로 해결
3. **Pretrained LLMs**: SAM/FastSAM/T-Rex2 같은 대형 사전학습 비전모델이 저조도에서도 의외로 잘 작동함(Fig.10) → 지식 증류 또는 저하 이미지 지식 주입 방향
4. **Multimodal Information**: 이벤트 카메라(140dB, 일반 카메라 60dB) 등 다른 센서 통합
5. **High-Dimensional Data**: 2D RGB를 넘어 3D 데이터(포인트클라우드, 터널/광산 등)로 확장
6. **Combining GSP and GNN**: 그래프 신호처리·그래프 신경망을 활용해 라벨 부족 상황에서도 강건한 모델 구축

---
*(2)편에서 이어서*