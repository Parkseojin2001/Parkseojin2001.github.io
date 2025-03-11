---
title: "11장 케라스를 사용한 인공 신경망 소개(2)"
excerpt: "옵티마이저/스케줄러/과대적합"

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

AdaGrad는 너무 빨라 느려져서 전역 최적점에 수렴하지 못하는 위험을 `RMSProp` 알고리즘으로 가장 최근 반복에서 비롯된 그레이디언트만 누적함으로써 이 문제를 해결했다.
  - 알고리즘의 첫 번째 단계에서 지수 감소를 사용한다.

**RMSProp 알고리즘**
1. $s \leftarrow \beta s + (1 - \beta)\nabla_\theta J(\theta) \otimes \nabla_\theta J(\theta)$
2. $\theta \leftarrow \theta - \eta \nabla_\theta J(\theta) \oslash \sqrt{s + \epsilon}$ 

```python
from tensorflow.keras.optimizers import RMSProp

optimizer = RMSProp(lr=0.001, rho=0.9) # rho는 베터에 해당(default=0.9)
```
- 간단한 문제를 제외하고는 RMSProp 옵티마이저가 훨씬 성능이 좋았으며 Adam 최적화가 나오기 전까지 가장 선호하는 최적화 알고리즘이었다.

## 11.3.5 Adam과 Nadam 최적화

`Adam`은 **적응적 모멘트 추정**을 의미한다.
- 모멘트 최적화(지난 그레이디언트의 지수 감소 평균을 따름)와 RMSProp(지난 그레이디언트 제곱의 지수 감소된 평균을 따름)의 아이디어를 합침

**Adam 알고리즘**<br>
1. $m \leftarrow \beta_1 m - (1 - \beta_1)\nabla_\theta J(\theta)$
2. $s \leftarrow \beta_2 s + (1 - \beta_2)\nabla_\theta J(\theta) \otimes \nabla_\theta J(\theta)$
3. $\hat m \leftarrow \frac{m}{1 - \beta_1^t}$
4. $\hat s \leftarrow \frac{s}{1 - \beta_2^t}$
5. $\theta \leftarrow \theta + \eta \hat m \oslash \sqrt{\hat s + \epsilon}$
  - $\beta_1$ : 모멘텀 감쇠 하이퍼파라미터(default=0.9)
  - $\beta_2$ : 스케일 감쇠 하이퍼파라미터(default=0.999)

```python
from tensorflow.keras.optimizers import Adam

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
```
- Adam이 적응적 학습률 알고리즘이기 때문에 학습률 파라미터 $\eta$(default=0.001)를 튜닝할 필요가 적다.

**Adam 변종**<br>
- AdamMax
  - 실전에서 AdaMax가 Adam보다 안정적이지만 데이터셋에 따라 다르고 일반적으로 Adam 성능이 낫다.
  - 어떤 작업에서 Adam이 잘 동작하지 않는다면 시도할 수 있는 옵티마이저 중 하나

```python
from tensorflow.keras.optimizers import Adamax

optimizer = Adamax(lr=0.001, beta_1=0.9, beta_2=0.999)
```

- Nadam
  - Adam 옵티마이저 + 네스테로프 기법으로 종종 Adam보다 조금 더 빠르게 수렴
  - 일반적으로 Adam보다 성능이 좋지만 경우에 따라 RMSProp이 더 좋기도 함

```python
from tensorflow.keras.optimizers import Adamax

optimizer = Adamax(lr=0.001, beta_1=0.9, beta_2=0.999)
```

> 적응적 최적화 방법(RMSProp, Adam, Nadam)이 좋은 솔루션으로 빠르게 수렴하지만 일부 데이터셋에서 나뿐 결과를 낸다. 이런 경우에는 기본 네스테로프 가속 경사를 사용할 수 있다.

위의 모든 최적화 기법은 **1차 편미분(야코비안)**에만 의존한다. **2차 편미분(헤시안)**을 기반으로 한 알고리즘이 있지만 메모리 용량을 넘어서는 문제와 계산 속도가 느리기때문에 심층 신경망에 적용하기 어렵다.

**옵티마이저 비교**(`*` = 나쁨, `**` = 보통, `***` = 좋음)<br>
- 선택한 옵티마이저의 성능이 만족스럽지 않을 경우 기본 `Nesterov` 가속 경사 사용 추천

|클래스|수렴 속도|수렴 품질|
|----|-------|-------|
|SGD|`*`|`***`|
|SGD(momentum=...)|`**`|`***`|
|SGD(momentum=..., nesterov=True)|`**`|`***`|
|Adagrad|`***`|`*`(너무 일찍 멈춤)|
|RMSprop|`***`|`**` 또는 `***`|
|Adam|`***`|`**` 또는 `***`|
|Nadam|`***`|`**` 또는 `***`|
|AdaMax|`***`|`**` 또는 `***`|

## 11.3.6 학습률 스케줄링

학습률 선택이 훈련의 성패를 가른다.

<img src="https://user-images.githubusercontent.com/78655692/148238088-308988e0-92f9-495c-90db-8e3da6d2ea6f.png">

큰 학습률로 시작하고 학습 속도가 느려질 때 학습률을 낮추면 최적의 고정 학습률보다 좋은 솔루션을 더 빨리 발견할 수 있다. 훈련하는 동안 학습률을 감소시키는 전략 중 **학습 스케줄**이 있다.

- 거듭제곱 기반 스케줄링
  - 학습률을 반복 횟수에 t에 대한 함수 $\eta(t) = \eta_0 / {(1 + t/s)^c}$로 지정
  - $t = k \cdot s$로 커지면 학습률이 $\frac{\eta_0}{k + 1}$로 줄어듦.
  - 하이퍼파라미터
    - $\eta_0$ : 초기 학습률
    - $c$ : 거듭제곱수, 일반적으로 1로 지정
    - $s$ : 스텝 횟수

```python
from tensorflow.keras.optimizers import SGD

optimizer = SGD(lr=0.01, decay=1e-4)
```

- 지수 기반 스케줄링
  - $\eta(t) = \eta_0 \ 0.1^{t/s}$
  - 학습률이 $s$ 스텝마다 10배씩 점차 줄어든다.

```python
def exponential_decay_fn(epoch): # 현재 에포크의 학습률을 받아 반환하는 함수 필요
    return 0.01 * 0.1 ** (epoch / 20)

def exponential_decay_fn(epoch, lr):  # 현재 학습률을 매개변수로 받음
    return lr * 0.1 ** (1 / 20)

# 변수를 설정한 클로저를 반환하는 방식
def exponential_decay(lr0, s):
  def exponential_decay_fn(epoch):
    return lr0 * 0.1**(epoch / s)

exponential_decay_fn = exponential_decay(lr0=0.01, s=20)
```

스케줄링 함수를 전달하여 `LearningRateScheduler` 콜백을 만들고 이 콜백을 `fit()` 메서드에 전달한다.

```python
from tensorflow.keras.callbacks import LearningRateScheduler

lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)
history = model.fit(X_train_scaled, y_train, [...], callbacks=[lr_scheduler])
```


- 구간별 고정 스케줄링
  - 일정 횟수의 에포크 동안 일정한 학습률을 사용하고 그 다음 또 다른 횟수의 에포크 동안 작은 학습률을 사용한다.
  - 학습률

  ```python
  def piecewise_constant_fn(epoch):
    if epoch < 5:
        return 0.01
    elif epoch < 15:
        return 0.005
    else:
        return 0.001
  ```

  - 콜백함수 지정

  ```python
  from tensorflow.keras.callbacks import LearningRateScheduler

  lr_scheduler = LearningRateScheduler(piecewise_contant_fn)
  history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                      validation_data=(X_valid_scaled, y_valid),
                      callbacks=[lr_scheduler])
  ```

- 성능 기반 스케줄링
  - 매 $N$ 스텝마다 검증 오차를 측정하고 오차가 줄어들지 않으면 $\lambda$배만큼 학습률을 감소시킨다.
  - `ReduceLROnPlateau` 콜백 클래스 활용

```python
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.schedules import ExponentialDecay
from tensorflow.kear.optimizers import SGD

lr_scheduler = ReduceLROnPlateau(factor=0.5, patience=5)

s = 20 * len(X_train) // 32
learning_rate = ExponentialDecay(0.01, s, 0.1)
optimizer = SGD(learning_rate=learning_rate)
```
- 1 사이클 스케줄링
  - 학습률을 훈련 과정 중에 올리거나 내리도록 조정
    - 훈련 전반부: 낮은 학습률 $\eta_0$에서 높은 학습률 $\eta_1$까지 선형적으로 높힘
    - 훈련 후반부: 다시 선형적으로 $\eta_0$까지 낮춤
    - 훈련 마지막 몇 번의 에포크: 학습률을 소수점 몇 자리까지 선형적으로 낮춤


**결론**<br>
- 지수 기반 스케줄링, 성능 기반 스케줄링, 1 사이클 스케줄링이 수렴 속도를 크게 높일 수 있다.
- 성능 기반과 지수 기반 모두 좋지만 **지수 기반 스케줄링**을 선호
  - 높은 성능, 튜닝과 구현이 쉬음


# 11.4 규제를 사용해 과대적합 피하기

## 11.4.1 $\ l_1$ 과 $\ l_2$ 규제

## 11.4.2 드롭아웃

## 11.4.3 몬테 카를로 드롭아웃

## 11.4.4 맥스-노름 규제

# 11.5 요약 및 실용적인 가이드라인
