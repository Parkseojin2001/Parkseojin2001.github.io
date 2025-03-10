---
title: "11장 케라스를 사용한 인공 신경망 소개(2)"
excerpt: "옵티마이저/과대적합"

categories:
  - 핸즈온 머신러닝
tags:
  - [hands-on]

permalink: /hands-on/DNN-2/

toc: true
toc_sticky: true
math: true

date: 2025-03-07
last_modified_at: 2025-03-07
---


# 11.3 고속 옵티마이저

**훈련 속도를 높이는 방법**
- 연결 가중치에 좋은 초기화 전략 사용하기
- 좋은 활성화 함수 사용하기
- 배치 정규화 사용하기
- 사전훈련 네트워크 일부 재사용

훈련 속도를 크게 높일 수 있는 또 다른 방법으로 표준적인 경사 하강법 옵티마이저 대신 더 빠른 옵티마이저를 사용할 수 있다.
- 모멘텀 최적화
- 네스테로프(Nesterov) 가속 경사(NAG)
- AdaGrad
- RMSProp
- Adam 최적화
- Nadam 최적화

<img src="https://oopy.lazyrockets.com/api/v2/notion/image?src=https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Fd72c4064-00fd-4e81-984e-cf5f4a98340e%2FUntitled.png&blockId=1248bc15-77fa-4ba8-954c-a57962abe99f">

## 11.3.1 모멘텀 최적화
표준적인 경사 하강법은 경사면을 따라 일정한 크기의 스텝으로 조금씩 내려간다.
- 경사 하강법 공식 : $\theta \leftarrow \theta - \eta \nabla_\theta J(\theta)$
  - $\theta$ : 가중치
  - $\eta$ : 학습률
  - $J(\theta)$ : 가중치에 대한 비용 함수
- 이 식은 이전 그레이디언트가 얼마였는지 고려하지 않는다.

모멘텀 최적화는 이전 그레이디언트가 얼마였는지를 상당히 중요하게 생각한다. 
- 매 반복에서 현재 그레이디언트를 (학습률 $\eta$를 곱한 후) **모멘텀 벡터 m**에 더하고 이 값을 빼는 방식으로 가중치를 갱신한다.
  - 그레이디언트를 가속도롤 사용한다.
  - 일종의 마찰저항을 표현하고 모멘텀이 너무 커지는 것을 막기 위해 **모멘텀**이라는 새로운 하이퍼파라미터 $\beta$를 사용
    - 0(높은 마찰 저항)과 1(마찰 저항 없음) 사이로 설정되어야 한다. (default = 0.9)

**모멘텀 알고리즘**<br>

1. $m \leftarrow \beta m - eta \nabla_\theta J(\theta)$
2. $\theta \leftarrow \theta + m$

<img src="https://user-images.githubusercontent.com/78655692/147542303-8f8b631e-b95f-4e35-9deb-b524c3f03948.png">

- 모멘텀 최적화는 골짜기를 따라 바닥(최적점)에 도달할 때까지 점점 더 빠르게 내려간다.
- 모멘텀 최적화를 사용하면 지역 최적점(local optimia)을 건너 뛰는데 도움을 준다.

> 모멘텀떄문에 옵티마이저가 최적값에 안정되기 전까지 건너뛰었다가 다시 돌아오고, 다시 또 건너뛰는 식으로 여러 번 왔다 갔다 할 수 있다.

```python
from tensorflow.keras.optimizers import SGD

optimizer = SGD(lr=0.001, momentum=0.9)
```

## 11.3.2 네스테로프 가속 경사

네스테로프 가속 경사(NAG: Nesterov accelerated gradient)는 현재 위치가 $\theta$가 아니라 모멘텀의 방향으로 조금 앞선 $\theta + \beta m$에서 비용 함수의 그레이디언트를 계산한다.
- 모멘텀 최적화의 변종으로 거의 항상 빠르다.

**네스테로프 가속 경사 알고리즘**<br>

1. $m \leftarrow \beta m - \eta \nabla_\theta J(\theta + \beta m)$
2. $\theta \leftarrow \theta + m$

<img src = "https://user-images.githubusercontent.com/78655692/147552717-982cc6ea-1e0f-41fc-82a7-a94e1d5a8ea1.png"  height = "600px" width = "500px">

- 기본 모멘텀 최적화는 시작점에서 그레이디언트를 사용하고 네스테로프 가속 경사는 그 방향으로 조금 더 나아가서($\bata m$) 그레이디언트를 측정하고 이것을 사용한다.
  - 모멘텀은 골짜기를 가로지르는($\eta \nabla_1$) 반면에 NAG는 계곡의 아래쪽($\eta \nabla_2$)으로 잡아당기게 되며 이는 진동을 감소시키고 수렴을 빠르게 만들어준다.

```python
optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
```

## 11.3.3 AdaGrad

경사 하강법은 전역 최적점 방향으로 곧장 향하지 않고 가장 가파른 경사를 따라 빠르게 내려가기 시작해서 골짜기 아래로 느리게 이동한다. 이 문제를 보완하기 위해서 `AdaGrad` 알고리즘이 고안되었다.
- 가장 가파른 차원을 따라 그레이디언트 벡터의 스케일이 감소된다.

**AdaGrad 알고리즘**<br>
1. $s \leftarrow s + \nabla_\theta J(\theta) \otimes \nabla_\theta J(\theta)$
2. $\theta \leftarrow \theta - \eta \nabla_\theta J(\theta) \oslash \sqrt{s + \epsilon}$

이 알고리즘은 학습률을 감소시키지만 경사가 완만한 차원보다 가파른 차원에 대해 더 빠르게 감소되며 이를 **적응적 학습률**이라고 부른다.

<img src="https://user-images.githubusercontent.com/78655692/147553505-2a077b07-0fa7-4769-ba10-24b1a359d933.png" height = "350px" width="500px">

장점<br>
  - 전역 최적점 방향으로 더 곧장 가도록 갱신되는 데 도움이 된다.
  - 학습률 하이퍼파라미터 $\eta$를 덜 튜닝해도 된다.
  - 2차방정식 문제에 잘 작동한다.

단점<br>
  - 학습률이 너무 감소되어 전역 최적점에 도착하기 전에 알고리즘이 완전히 멈춘다.
  - 심층 신경망에는 사용하지 않아야 하며 선형 회귀 같은 간단한 작업에는 효과적이다.

## 11.3.4 RMSProp

## 11.3.5 Adam과 Nadam 최적화

## 11.3.6 학습률 스케줄링


# 11.4 규제를 사용해 과대적합 피하기

## 11.4.1 $\ l_1$ 과 $\ l_2$ 규제

## 11.4.2 드롭아웃

## 11.4.3 몬테 카를로 드롭아웃

## 11.4.4 맥스-노름 규제

# 11.5 요약 및 실용적인 가이드라인
