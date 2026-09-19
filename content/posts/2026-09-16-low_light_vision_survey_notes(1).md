---
title: "Deep Learning for Low-Light Vision: A Comprehensive Survey"
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

## Section IV. Other High-Level Low-Light Vision Tasks

### IV-A. Semantic Segmentation
- 낮 세그멘테이션은 잘 되지만 야간 등 조명 불량 상황에서 기존 모델의 일반화 성능이 떨어짐
- **Dark Zurich** : 낮/황혼/밤 정렬 이미지 데이터셋 제안 (uncertainty-aware 프레임워크)
- **ACDC** : Dark Zurich 후속으로 나온 대규모 악조건(안개/밤/비/눈) 데이터셋
- Retinex 기반 향상 + 세그멘테이션 결합, 향상·인식 동시 처리하는 cascaded 구조도 언급됨

### IV-B. Object Tracking
- UAV(드론) 추적에서 저조도로 인해 추적이 끊기는 문제를 다룸
- **DarkLighter** : Retinex 기반 향상으로 전처리 후 UAV 트래커와 함께 최적화
- 후속 연구: Transformer 기반 향상기+트래커를 task-driven 방식으로 결합
- 이벤트 카메라+일반 카메라, 적외선+RGB 등 다른 센서 결합으로 저하 상황에서도 추적 유지

### IV-C. Human Pose Estimation
- 강한 조명·그림자는 보통 방해 요소지만, 오히려 **그림자를 보조 카메라처럼 활용** 해 자세·형태를 복원하는 연구도 있음 (Balan et al.)
- 반대로 저조도는 노이즈·낮은 대비로 자세 추정이 더 어려워짐
- **ExLPose** : 저조도 전용 자세 추정 데이터셋+모델 (Lee et al.)
- UIRE-Net: 비지도 방식의 조도 반사율 추정으로 야간 자세 추정 지원

### IV-D. Moving Object Detection (배경 분리)
- **배경 제거(background subtraction)** 기법으로 움직이는 물체를 찾아내는 방식
- 조명 변화·그림자·날씨 변화 등 복잡한 환경에서 정확도가 떨어지는 게 문제
- 시각적 주의 메커니즘 + 자기조직화 신경망 결합, zero-shot 배경 모델링 등으로 강건성 개선 시도

## Section V. Datasets and Evaluation Metrics for Low-Light Vision

### V-A. Datasets Overview
저조도 데이터셋을 **라벨(annotation) 유무** 기준으로 두 그룹으로 나눠 소개.

### V-B. Dataset Without Annotations (라벨 없음 — LLIE 화질 평가용)
| 데이터셋 | 설명 |
|---|---|
| MIT-Adobe FiveK | 5000장 RAW, 전문가 5명이 각각 보정한 버전 포함 |
| LIME/NPE/MEF/DICM | 8~64장 소규모 실제 저조도 테스트셋 |
| LOL (v1, v2) | 노출 조절로 찍은 저조도-정상광 페어 이미지 |
| SID | 5094장, 짧은/긴 노출 페어 (실내 0.03~0.3lux, 실외 0.2~5lux) |
| SICE | 589개 다중노출 시퀀스, HDR 레퍼런스 포함 |
| SMOID / DRV | RAW 비디오 페어 데이터셋 |
| VE-LOL-L | 합성 1000 + 실제 1500장 페어 |
| SDSD | 레일 카메라로 두 번 촬영한 150개 페어 비디오 |
| LLIV-Phone | 18개 스마트폰 기종, 45148장, 다양한 조명 |

### V-C. Dataset With Annotations (라벨 있음 — 탐지/세그멘테이션용)
| 데이터셋 | 설명 |
|---|---|
| Dark Zurich | 낮/황혼/밤 이미지, 일부 픽셀 단위 세그멘테이션 라벨 |
| ExDARK | 실내외 10단계 조도, 클래스 라벨 + 바운딩박스 |
| DARK FACE | 1만 장, 얼굴 바운딩박스 |
| LLVIP | 가시광+적외선 페어, 보행자 라벨, 시공간 정렬 |
| VE-LOL-H | 1만940장, 얼굴 라벨 |
| ACDC | 4006장, 안개/밤/비/눈, 픽셀 단위 시맨틱 라벨 |
| DarkVision | 정적/동적 분리, 조도 5단계, 대량 바운딩박스 |

### V-D. Image Quality Evaluation Metrics (화질 평가지표)
- **Full-Reference (정답 필요)**: PSNR(↑, 픽셀 오차 기반), MAE(↓, 평균 오차), SSIM(↑, 구조·명암·대비 유사도), LPIPS/DISTS(↓, 딥러닝 기반 지각 유사도)
- **No-Reference (정답 불필요)**: NIQE(↓, 자연 이미지와의 통계적 거리), LOE(↓, 밝기 순서 왜곡)
- **LLM 기반**: Q-Bench 등, LLM이 이미지 품질을 언어로 평가·점수화하는 최신 흐름

### V-E. Recognition Accuracy Evaluation Metrics (인식 성능 평가지표)
- **객체 탐지**: AP, mAP(전체 클래스 평균), AP50/AP75(IoU 임계값 0.5/0.75)
- **세그멘테이션**: mIoU, MPA(클래스별 픽셀 정확도), Dice coefficient, Boundary F1

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