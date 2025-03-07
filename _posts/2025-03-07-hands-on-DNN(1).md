---
title: "11장 케라스를 사용한 인공 신경망 소개(1)"
excerpt: "그레이디언트/전이학습/비지도학습"

categories:
  - 핸즈온 머신러닝
tags:
  - [hands-on]

permalink: /hands-on/DNN-1/

toc: true
toc_sticky: true
math: true

date: 2025-03-07
last_modified_at: 2025-03-07
---

데이터에 따라 깊은 심층 신경망을 훈련해야 한다. 심층 신경망을 훈련하는 도중에 다음과 같은 문제를 마주할 수 있다.

- **그레이디언트 소실** 또는 **그레이디언트 폭주** 문제에 직면할 수 있다. 심층 신경망의 아래쪽으로 갈수록 그레이디언트가 점점 더 작아지거나 커지는 현상이다. 두 현상 모두 하위층을 훈련하기 매우 어렵게 만든다.
- 대규모 신경망을 위한 훈련 데이터가 충분하지 않거나 레이블을 만드는 작업에 비용이 너무 많이 들 수 있다.
- 훈련이 극단적으로 느려질 수 있다.
- 수백만 개의 파라미터를 가진 모델은 훈련 세트에 과대적합될 위험이 매우 크며 특히 훈련 샘플이 충분하지 않거나 잡음이 많은 경우 발생한다.

# 11.1 그레이디언트 소실과 폭주 문제

**그레이디언트 소실**<br>
알고리즘이 하위층으로 진행될수록 그레이디언트가 점점 작아지는 경우가 많으며 이는 하위층의 연결 가중치를 변경되지 않은 채로 두게되며 훈련이 좋은 솔루션으로 수렴되지 않는 현상

**그레이디언트 폭주**<br>
그레이디언트가 점점 커져서 여러 층이 비정상적으로 큰 가중치로 갱신되어 알고리즘이 발산할 수 있다. 주로 순환 신경망에서 나타난다.

원인은 로지스틱 시그모이드 활성화 함수와 가중치 초기화 방법의 조합이었다. 
- 각 층에서 출력의 분산이 입력의 분산보다 더 크다.
- 신경망의 위쪽으로 갈수록 층을 지날 때마다 분산이 계속 커져 가장 높은 층에서는 활성화 함수가 0이나 1로 수렴한다.
- 로지스틱 함수의 평균이 0이 아니고 0.5라는 사실 때문에 더 나빠진다.

<img src="https://user-images.githubusercontent.com/78655692/148024675-70b17a10-cba9-480a-ab91-189d4722c19b.png" height="400px" width="500px">

- 입력이 0이나 1로 수렴해서 기울기가 0에 매우 가까워진다.
- 역전파가 될 때 전파할 그레이디언트가 거의 없고 조금 있는 그레이디언트는 최상위층에서부터 역전파가 진행되면서 점차 약해져서 실제로 아래쪽 층에는 아무것도 도달하지 않게 된다.

## 11.1.1 글로럿과 He 초기화
- 층에 사용되는 활성화 함수의 종류에 따라 세 가지 초기화 방식 중 하나를 선택
  - 글로럿(Glorot) 초기화
  - 르쿤(LeCun) 초기화
  - 헤(He) 초기화
- 예측을 할 때는 정방향으로, 그레이디언트를 역전파할 때는 역방향으로 양방향 신호가 적절하게 흘러야 한다.
- 글로럿과 벤지오는 적절한 신호가 흐르기 위해서는 각 충의 출력에 대한 분산이 입력에 대한 분산과 같아야 한다고 주장했다. 그리고 역방향에서 층을 통과하기 전과 후의 그레이디언트 분산이 동일해야 한다.

**세이비어 초기화** or **글로럿 초기화**<br>
$$fan_{avg} = \frac{fan_{in} + fan_{out}}{2}$$
- $fan_{in}$(팬-인) : 층에 들어오는 입력 수
- $fan_{out}$(팬-아웃) : 층에 들어오는 입력 수
- 평균($\mu$)이 0이고 분산($\sigma^2$)이 $\sigma^2 = \frac{1}{fan_{avg}}$ 인 정규분포
- $r = \sqrt{\frac{3}{fan_{avg}}}$ 일 때 $-r$과 $+r$ 사이의 균등분포

**르쿤 초기화**<br>
- 글로럿 초기화 정의에서 $fan_{avg}$를 $fan_{in}$으로 변경

**He 초기화**<br>
- ReLU와 ReLU의 변종 활성화 함수에 대한 초기화 전략

#### 활성화 함수와 초기화 방식

|초기화 전략|활성화 함수|$\sigma^2$(정규분포)|
|--------|--------|------------------|
|글로럿|활성화 함수 없음, 하이퍼볼릭 탄젠트, 로지스틱, 소프트맥스|$1/fan_{avg}$|
|He|ReLU 함수와 그 변종들|$2/fan_{in}$|
|르쿤|SELU|$1/fan_{in}$|

케라스는 기본적으로 균등분포의 글로럿 초기화를 사용한다. 다음과 같이 층을 만들 때 `kernel_initializer="he_uniform"` 이나 `kernel_initializer="he_normal"`로 바꾸어 He 초기화를 사용할 수 있다.

```python
from tensorflow.keras.layers import Dense

keras.layers.Dense(10, activation="relu", kernel_initializer="he_normal")
```

$fan_{in}$ 대신 $fan_{out}$ 기반의 균등분포 He 초기화를 사용하는 경우
- Variance Scaling 사용

```python
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Dense

he_avg_init = VarianceScaling(scale=2., mode='fan_avg',
                              distribution='uniform')
keras.layers.Dense(10, activation='sigmoid', kernel_initializer=he_avg_init)
```
## 11.1.2 수렴하지 않는 활성화 함수

그전에는 대부분 시그모이드 활성화 함수가 최선의 선택일 것이라고 생각했다. 하지만 ReLU 함수는 특정 양수값에 수렴하지 않으며 계산이 빠르다는 큰 장점이 있다.<br>
하지만 dying ReLU로 알려진 문제점이 있다.
- 훈련하는 동안 일부 뉴런이 0 이외의 값을 출력하지 않는다라는 의미

가중치 합이 음수이면 ReLU 함수의 그레이디언트가 0이 되므로 경사하강법이 더는 작동하지 않는다.

해결책: `LeakyReLU` 함수
- $LeakyReLU_{\alpha}(z) = max(\alpha z, z)$
- 하이퍼파라미터 $\alpha$가 이 함수가 '새는' 정도를 결정한다.
  - 새는 정도란 $z < 0$ 일 때 이 함수의 기울기이며, 일반적으로 0.01로 설정한다. 이 작은 기울기가 LeakyReLU를 절대 죽지 않게 만들어준다.
- ReLU보다 좋은 성능 발휘(default = $\alpha = 0.1$)
  - $\alpha = 0.2$로 할 때 좀 더 성능이 좋아짐

<img src="https://user-images.githubusercontent.com/78655692/148028052-8b6c9ff4-9e5b-44bc-9676-f317fe533aab.png" height = "400px" width="500px">

- `RReLU`(randomized leaky ReLU) : 훈련하는 동안 주어진 범위에서 $\alpha$를 무작위로 선택하고 테스트시에는 평균을 사용
  - 과대적합을 줄이는 규제역할도 수행
- `PReLU`(parametric leaky ReLU) : $\alpha$가 훈련하는 동안 학습(하이퍼파라미터가 아니고 다른 모델 파라미터와 마찬가지로 역전파에 의해 변경)
  - 대규모 이미지 데이터셋에서 ReLU보다 성능이 좋음
  - 소규모 데이터셋에서는 과대적합 위험성 존재
- `ELU`(exponential linear unit): 앞에서 언급된 ReLU 변종들보다 성능이 좋음
  - 훈련 시간이 줄고 신경망의 테스트 세트 성능도 더 높음

$$ELU_{\alpha}(z)=
\begin{cases}
\alpha(exp(z)-1) \; if\;z<0\\
z \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; if\;z\geq0
\end{cases}$$

<img src="https://user-images.githubusercontent.com/78655692/148224276-da25c5c6-1bf5-469c-93f7-ae6a1e3226bc.png" height="400px" width="500px">

- $z < 0$일 때 음수값이 들어오므로 활성화 함수의 평균 출력이 0에 더 가꿔워진다. 이는 그레이디언트 소실 문제를 완화해준다.
- 하이퍼파라미터 $\alpha$는 z가 큰 음수값일 때 ELU가 수렴할 값을 정의한다.
- $z < 0$이어도 그레이디언트가 0이 아니므로 죽은 뉴런을 만들지 않는다.
- $\alpha=1$이면 이 함수는 $z=0$에서 급격히 변동하지 않으므로 $z=0$을 포함해 모든 구간에서 매끄러워 경사 하강법의 속도를 높여준다.

**장단점**<br>
- 수렴 속도가 빠르다.
- 지수 함수를 사용하므로 ReLU나 그 변종들보다 계산이 느리다.
- 훈련 시에는 수렴 속도가 빨라서 느린 계산이 상쇄되지만 테스트 시에는 ELU를 사용한 네트워크기 ReLU를 사용한 네트워크보다 느릴 것이다.

- `SELU`(Scaled ELU) : ELU 활성화 함수의 변종
  - 완전 연결 층만 쌓아서 신경망을 만들고 모든 은닉층이 SELU 활성화 함수를 사용한다면 네트워크가 자기 정규화(self-normalized)된다고 주장
  - 훈련하는 동안 각 층의 출력이 평균 0과 표준편차 1을 유지하는 경향이 있다.
    - 그레이디언트 소실과 폭주 문제를 막아준다.
  - 다른 활성화 함수보다 뛰어난 성능을 종종 보이지만 자기 정규화가 일어나기 위한 몇 가지 조건이 있다.
    1) 입력 특성이 반드시 표준화(평균 0, 표준편차 1)되어야 한다.
    2) 모든 은닉층의 가중치는 르쿤 정규분포 초기화로 초기화되어야 한다. 케라스에서는 `kernel_initializer="lecun_normal`로 설정
    3) 네트워크는 일렬로 쌓은 층으로 구성되어야 한다. 순환 신경망이나 스킵 연결과 같이 순차적이지 않은 구조에서 사용하면 자기 정규화되는 것을 보장하지 않는다.
    4) 모든 층이 완전연결층이어야 한다.

#### 활성화 함수 쓰는 방법
- 일반적으로 SELU > ELU > LeakyReLU(그리고 변종들) > ReLU > tanh > sigmoid 순
- 네트워크가 자기 정규화되지 못하는 구조라면 SELU 보단 `ELU`
- 실행 속도가 중요하다면 `LeakyReLU`(하이퍼파라미터를 더 추가하고 싶지 않다면 케라스에서 사용하는 기본값 
$\alpha$ 사용)
- 시간과 컴퓨팅 파워가 충분하다면 교차 검증을 사용해 여러 활성화 함수를 평가
- 신경망이 과대적합되었다면 `RReLU`
- 훈련세트가 아주 크다면 `PReLU`
- ReLU가 가장 널리 사용되는 활성화 함수이므로 많은 라이브러리와 하드웨어 가속기들이 ReLU에 특화되어 최적화.
따라서 속도가 중요하다면 `ReLU`가 가장 좋은 선택

**`LeakyReLU` 활성화 함수 사용**<br>

```python
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.models import Sequential

model = Sequential([
  [...]
  Dense(10, kernel_initializer="he_normal"),
  LeakyReLU(alpha=0.2)
  [...]
])
```

**`PReLU` 활성화 함수 사용**<br>
- PReLU층을 만들고 모델에서 적용하려는 층 뒤에 추가

```python
from tensorflow.keras.layers import Flatten, Dense, PReLU
from tensorflow.keras.models import Sequential

model = Sequential([
  Flatten(input_shape=[28, 28]),
  Dense(300, kernel_initializer='he_normal'),
  PReLU(),
  Dense(100, kernel_initializer='he_normal'),
  PReLU(),
  Dense(10, activation='softmax')
])
```

**`SELU` 활성화 함수 사용**<br>
- 층을 만들 때 `activation='selu'` 와 `kernel_initializer='lecun_normal` 지정

```python
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Flatten(input_shape=[28, 28]))
model.add(Dense(300, activation='selu', kernel_initializer='lecun_normal'))

for layer in range(99):
  model.add(Dense(100, activation='selu', kernel_initializer='lecun_normal'))
model.add(Dense(10, activation='softmax'))
```

## 11.1.3 배치 정규화

## 11.1.4 그레이디언트 클리핑

# 11.2 사전훈련된 층 재사용하기

## 11.2.1 케라스를 사용한 전이 학습

## 11.2.2 비지도 사전훈련

## 11.2.3 보조 작업에서 사전훈련

