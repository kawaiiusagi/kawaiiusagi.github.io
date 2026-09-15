---
title: "Autonomous Vehicles Perception (AVP) Using Deep Learning: Modeling, Assessment, and Challenges"
date: 2026-09-15
draft: false
tags: ["Paper_Review", "Autonomous Driving", "Perception", "study-log"]
categories: ["Survey_Notes"]
---
| 원문 링크 : https://doi.org/10.1109/ACCESS.2022.3144407
- **제목** : Autonomous Vehicles Perception (AVP) Using Deep Learning: Modeling, Assessment, and Challenges
- **저자**: Hrag-Harout Jebamikyous, Rasha Kashef (Ryerson University)
- **분류**: Survey (자율주행 Perception)
- **읽은 범위**: Abstract, Introduction, Section II (Active and Passive Sensors), Section III (AVP 개요), Section IV (Semantic Segmentation 개념 + Literature Review), Section V (Object Detection 개념 + Literature Review)

## 한 줄 요약

> 자율주행 인지(Perception)의 두 핵심 태스크인 **Semantic Segmentation**(픽셀 단위 분류)과 **Object Detection**(바운딩 박스 기반 탐지)을 중심으로, 이미지·LiDAR 포인트클라우드 기반 딥러닝 방법론과 최신 연구 사례들을 정리한 로드맵 성격의 서베이.

## 왜 이 논문을 골랐나 (지난 Motion Planning과의 대조)

지난번에 읽은 [Motion Planning 서베이](../2026-09-14-motion_planning_survey_notes)가 Planning/Control 영역이었다면, 이번엔 파이프라인의 앞단인 **Perception** 영역을 골라 읽었다. 아직 분야를 탐색하는 과정이기에 다른 부분의 서베이 논문도 읽어보고 싶었다.

## 핵심 개념 정리

### Active Sensor vs Passive Sensor

- **Active sensor** (Radar, LiDAR, Sonar): 스스로 에너지(전파, 레이저)를 주변에 쏘고, 그것이 물체에 반사되어 돌아오는 반응을 측정해서 정보를 얻는 센서
- **Passive sensor** (Stereo/Monocular Camera): 스스로 에너지를 쏘지 않고, 주변에서 오는 빛을 그냥 받아들이기만 하는 센서

| 센서 | 장점 | 단점 |
|---|---|---|
| Monocular Camera | 형태·질감·색상 정보 풍부 (차선 색, 신호등 색, 표지판 등) | 깊이(거리) 정보 제공 불가 |
| Stereo Camera | Monocular의 깊이 정보 한계를 보완, 상대적 깊이 추정 가능 | - |
| LiDAR | 조명/날씨와 무관하게 정확한 거리·위치 측정, 초당 수천 개 펄스로 360도 포인트클라우드 생성 | 색상 인식 불가, 신호등 색/표지판 글자 인식 불가 → 단독 사용 불가, 항상 Camera와 결합 |
| Radar | 눈·안개·비 등 악천후에서 카메라·LiDAR보다 강함 | 정확도 낮음, 상세 정보 부족 → 좁은 용도로만 사용, Camera/LiDAR와 결합 |

→ 이 표가 결국 **멀티센서 퓨전(Multi-sensor Fusion)** 이 필요한 이유가 된다. 각 센서의 단점을 다른 센서로 보완하기 위해 여러 센서 데이터를 결합해서 사용한다고 한다.

### Spatial Feature vs Semantic Feature — 처음에 헷갈렸던 부분

- **Spatial feature (공간적 특징)**: "어디에, 어떤 형태로 있는가" — 이미지 모드에서는 이웃한 셀들의 묶음(region), 벡터 모드에서는 선(line)·점(point)·폴리곤(polygon)으로 표현됨. LiDAR 데이터에서도 얻을 수 있고 GIS(Geographic Information System)로 처리·분석됨
  - Polygon은 3개 이상의 선분으로 둘러싸인 평면 도형을 말한다.
- **Semantic feature (의미론적 특징)**: "그것이 무엇을 의미하는가" — 이미지의 저수준 특징(색 등)을 실제 의미와 연결하는 것. 예: 초록색 = 나무, 파란색 = 하늘/바다. 자율주행에서는 차량, 도로표지판, 신호등, 차선 표시 등이 semantic feature에 해당

→ **Semantic Segmentation**이라는 이름 자체가 "픽셀에 의미(semantic)를 부여하는 분할(segmentation) 작업"이라는 뜻으로, 이 두 feature 개념이 챕터 제목과 직접 연결된다는 걸 이해하고 나니 훨씬 명확해졌다.

### 원문으로 다시 살펴보자.

#### Semantic Segmentation (의미론적 분할)

> "the process of assigning each pixel in an image to a particular class"

이미지의 **각 픽셀**에 클래스(사람, 자전거, 나무 등)를 할당하는 작업. 픽셀 단위 이미지 분류(image classification at a pixel level)로 볼 수 있다. 같은 클래스의 픽셀들은 같은 색으로 칠해짐 (차량=빨강, 식물=초록, 건물=회색 등).

#### Object Detection과 Bounding Box

> "the task of identifying and locating an object of interest in an image and drawing a **bounding box** around that object"

**Bounding box**란 감지된 물체를 감싸는 사각형 박스를 뜻한다. Semantic Segmentation이 픽셀 하나하나를 정교하게 분류하는 것이라면, Object Detection은 물체를 사각형 박스로 대충 감싸서 "이 위치에 이런 물체가 있다"는 것만 빠르게 표시하는 방식이다. 이 방법은 Semantic Segmentation보다 정밀도가 낮은 대신 "구조적인" 속도가 빠르다.

### Point Cloud (포인트 클라우드)

3D 공간에 찍힌 **점들의 집합**으로, 아직 "이게 뭔지" 분류되기 전의 원본(raw) 데이터. 각 점은 (X, Y, Z) 좌표, RGB 색상값, 밝기(luminance) 정보를 가진다. 레이저 스캐너가 레이저 펄스를 쏘고 반사되어 돌아오는 시간을 측정해서 정확한 위치와 형태를 계산하며, 자율주행에서는 주로 LiDAR로 수집된다.

## 헷갈렸던 점들

**Q. Bounding box가 뭔가?**

→ 물체를 감지했을 때 그 물체를 감싸는 사각형 박스. Object Detection의 결과물을 표현하는 방식.

**Q. Spatial feature는 알겠는데 Semantic feature는 뭔지 모르겠다.**

→ 처음엔 이 정의 문장 자체를 놓치고 읽었다. 다시 찾아보니 "이미지의 색 등 저수준 정보를 실제 의미(나무, 하늘 등)와 연결하는 것"이라는 명확한 정의가 있었다. 이번 일로, 앞으로는 챕터 제목과 직접 관련된 용어 정의 문장은 속독 중에도 놓치지 않도록 유의해야겠다고 느꼈다.

**Q. Point cloud를 "물체를 인식한 걸 모아둔 것"이라고 이해했는데?**

→ "인식"이라는 표현이 부정확했다. Point cloud는 아직 AI가 "이게 사람이다/차다"라고 분류하기 **이전의 원본(raw) 측정 데이터**이고, 그 분류(인식)는 이 point cloud를 입력으로 받아 Object Detection/Segmentation 모델이 나중에 수행하는 별도의 작업이다.

## Literature Review 요약 (Section IV-B, V-B)

두 섹션 모두 개별 논문 사례를 나열하는 구조로, 하나의 스토리라인이라기보다 "이런 문제를 이런 방법으로 풀었다"는 사례들의 집합에 가까웠다.

### Semantic Segmentation 관련 연구가 겨냥하는 3가지 방향
1. 다양한 환경(날씨·조명)에서도 안정적인 성능 유지
2. 물체 경계(edge)를 더 정확하게 구분
3. 실시간 처리를 위한 연산량 감소
{{< details summary="연산량을 줄이는 방법 (예: [10])" >}}
매 프레임 화면 전체를 분석하는 대신, **각 프레임에서 1픽셀 두께의 가로선 하나만 샘플링한다.** 이 선들을 시간 순서대로 이어붙이면 도로를 위에서 훑은 듯한 하나의 이미지(road profile image)가 만들어지고, 이를 기반으로 Segmentation을 수행한다. **분석할 데이터양 자체가 줄어들어** 차량 주행 속도를 따라갈 수 있는 실시간 처리가 가능해진다.
{{< /details >}}

### Object Detection 관련 연구가 겨냥하는 방향
- 실시간 분류 속도 향상 (예: LiDAR 기반 Real AdaBoost, 0.07ms 분류)
- 움직이는 물체를 제거해 더 정확한 지도 생성 (YOLOv2로 차량 검출 후 LiDAR 프레임에서 제거)
- 주변 차량 감지 + 충돌 경고 (YOLOv2 기반 다중 클래스 검출)
- 탐지와 추적(tracking)의 결합 (YOLOv3 + Median Flow/Correlation tracking)

→ 공통적으로 **YOLO 계열 모델(YOLOv2, YOLOv3)** 이 표준처럼 자주 등장하며, LiDAR와 카메라 데이터를 결합하는 방향으로 연구가 발전하고 있다는 흐름을 확인할 수 있었다.

## 기존 지식과의 연결

- 지난 Motion Planning 서베이에서 스킵했던 "실제 알고리즘 구현"과 달리, 이 논문의 Literature Review는 YOLO, Faster-RCNN 같은 실제 사용되는 딥러닝 모델 이름이 구체적으로 등장해서, 파이토치로 학습 중인 딥러닝 지식과 직접 연결되는 느낌을 받았다.

## 총평

Motion Planning 서베이가 "왜 이 문제가 계산적으로 어려운가"를 증명하는 이론 중심이었다면, 이 Perception 서베이는 "실제로 어떤 모델이 어떤 문제를 어떻게 개선했는가"를 사례 중심으로 다뤄서 훨씬 실전적이고 딥러닝 실습(파이토치)과 맞닿아 있다는 느낌을 받았다. 다만 그만큼 등장하는 논문 수도 많고 개별 사례가 나열식이라, 하나하나 깊게 파기보다는 "이 분야가 겨냥하는 문제 유형이 무엇인지" 패턴을 잡는 방식으로 읽는 방식으로 진행했다.

이번 회독을 통해 확실히 느낀 건, Perception 분야는 이론적 난이도보다 **최신 모델 트렌드를 계속 따라가야 하는 속도감**이 중요한 분야라는 점이라고 느꼈다. 매년 새로운 아키텍처(ENet, SegNet, YOLOv2/v3, Faster-RCNN 등)가 쏟아지고, 그 각각이 특정 문제(경계 인식, 실시간성, 악천후 대응)를 해결하려는 시도라는 것을 알게 됐다. 

이 분야에서 연구하려면 이론적 깊이보다 **최신 논문과 벤치마크를 꾸준히 트래킹하고, 실제로 모델을 돌려보며 성능을 비교하는 실전 감각**이 중요한 능력일 것 같다는 생각이 들었다.

또한 이번에 용어 정의 문장(Semantic feature 등)을 속독 중에 놓쳤던 경험을 통해, 앞으로는 챕터 제목과 직접 연관된 핵심 정의 문장은 속독 단계에서도 표시해두고 넘어가야겠다는 교훈을 얻었다.

## 다음에 더 볼 것

- Section VI (Deep Learning for AVP) — CNN 기본 구조, Supervised/Reinforcement Learning 기반 세부 모델들
- Section VII~VIII (데이터셋, 평가지표) — 필요시 재방문
- End-to-end / Learning-based 자율주행 서베이 (지난 Motion Planning 노트에서 예고한 항목)
- 오늘 다룬 개념(Semantic Segmentation, Object Detection) 기반의 간단한 파이토치 프로젝트 아이디어 구체화
