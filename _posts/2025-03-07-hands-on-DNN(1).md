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

`ELU(or ReLU 변종) + He 초기화`를 사용하면 훈련 초 단계서 그이언트 소실이나 폭주 문제를 크게 감소 시킬 수 있지만, 훈련하는 동안 다시 발생할 수 있다.

**배치정규화**(BN; batch normalization)는 그레이디언트 소실과 폭주 문제를 해결하기 위해 등장했다.
- 활성화 함수를 통과하기 전이나 후에 모델에 연산을 하나 추가한다.
- 단순하게 입력을 원점에 맞추고 정규화한 다음, 각 층에서 두 개의 새로운 파라미터로 결과값의 스케일을 조정하고 이동시킨다.
  - 평균: 0으로 조정
  - 분산: 스케일 조정
- 하나는 스케일 조정에, 다른 하나는 이동에 사용한다.
- 대부분 신경망의 첫 번째 층으로 배치 정규화를 추가하면 훈련 세트를 표준화할 필요가 없다.
- 입력 데이터를 원점에 맞추고 정규화하려는 알고리즘은 평균과 표준편차를 추정해야하므로 현재 미니배치에서 입력의 평균과 표준편차를 평가한다.

<img src="https://user-images.githubusercontent.com/78655692/163700923-e5387adf-2460-43f1-bb11-58aba432abbc.png" height="350px" width="500px">

**배치 정규화 알고리즘**<br>

1. $\quad \mu_{\beta} = \frac{1}{m_{\beta}} \sum_{i=1}^{m_{\beta}} x^i$

2. $\quad \sigma_B^2 = \frac{1}{m_{\beta}} \sum_{i=1}^{m_{\beta}} (x^{(i)} - \mu_B)^2$

3. $\quad \hat{x}^{(i)} = \frac{x^{(i)} - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$

4. $\quad z^{(i)} = \gamma \otimes \hat{x}^{(i)} + \beta$

- 훈련하는 동안 배치 정규화는 입력을 정규화한 다음 스케일을 조정하고 이동시킨다.
- 훈련이 끝난 후 전체 훈련 세트를 신경망에 통과시켜 배치 정규화 층의 각 입력에 대한 평균과 표준편차를 계산한다.
  - 예측할 때 배치 입력 평균과 표준 편차로 이 '최종' 입력 평균과 표준편차를 대신 사용할 수 있다.
- 대부분 배치 정규화 구현은 층의 입력 평균과 표준편차의 이동 평균을 사용해 훈련하는 동안 최종 통계를 추정한다.
- 케라스의 `BatchNormalization` 층은 이를 자동으로 수행한다.
- 배치 정규화 층마다 네 개의 파라미터 벡터가 학습된다.
  - $\gamma$(출력 스케일 벡터)와 $\beta$(출력 이동 벡터)는 일반적인 역전파를 통해 학습된다.
  - $\mu$(최종 입력 평균 벡터)와 $\sigma$(최종 입력 표준편차 벡터)는 지수 이동 평균을 사용하여 추정된다.
    - $\mu$와 $\sigma$는 훈련하는 동안 추정되지만 훈련이 끝난 후에 사용된다.

결과적으로 심층 신경망에서 배치 정규화가 성능을 크게 향상시켰다.
- 그레이디언트 소실/폭주 문제가 크게 감소하여 하이퍼볼릭 탄젠트 또는 로지스틱 활성화 함수를 사용할 수 있다.
- 가중치 초기화에 네트워크가 덜 민감
- 훨씬 큰 학습률을 사용하여 학습 과정의 속도를 크게 높을 수 있다.
- 규제와 같은 역할을 하여 다른 규제 기법의 필요성을 줄여준다.

하지만, 배치 정규화를 사용할 때 에포크마다 더 많은 시간이 걸리므로 훈련이 느려지지만 더 적은 에포크로 동일한 성능에 도달할 수 있어 실제 걸리는 시간은 보통 더 짧다.

### 케라스로 배치 정규화 구현하기

은닉층의 활성화 함수 전이나 후에 `BatchNormalization` 층을 추가하면 된다.
- 모델의 첫 번째 층으로 배치 정규화를 추가할 수 있다.

```python
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential

model = Seqnetial([
  Flatten(input_shape=[28, 28]),
  BatchNormalization(),
  Dense(300, activation='elu', kernel_initializer="he_normal"),
  BatchNormalization(),
  Dense(100, activation="elu", kernel_initializer="he_normal"),
  BatchNormalization(),
  Dense(10, activation="softmax")
])
```

```python
model.summary()

# Model: "sequential_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# flatten_1 (Flatten)          (None, 784)               0         
# _________________________________________________________________
# batch_normalization_1 (Batch (None, 784)               3136      
# _________________________________________________________________
# dense_1 (Dense)              (None, 300)               235500    
# _________________________________________________________________
# batch_normalization_2 (Batch (None, 300)               1200      
# _________________________________________________________________
# dense_2 (Dense)              (None, 100)               30100     
# _________________________________________________________________
# batch_normalization_3 (Batch (None, 100)               400       
# _________________________________________________________________
# dense_3 (Dense)              (None, 10)                1010      
# =================================================================
# Total params: 271,346
# Trainable params: 268,978
# Non-trainable params: 2,368
# _________________________________________________________________
```

- 배치 정규화 층은 입력마다 네 개의 파라미터 $\gamma$, $\beta$, $\mu$, $\sigma$를 추가한다. (784 x 4 = 3,316개의 파라미터)
- 마지막 2개의 파라미터 $\mu$와 $\sigma$는 이동 평균이다. 이 파라미터는 역전파로 학습되지 않기 때문에 케라스는 'Non-trainable' 파라미터로 분류한다.

활성화 함수 전에 배치 정규화 층을 추가하려면 은닉층에서 활성화 함수를 지정하지 말고 배치 정규화 층 뒤에 별도의 층으로 추가해야한다. 또한 배치 정규화 층은 입력마다 이동 파라미터를 포함하기 때문에 이전 층에서 편향을 뺄 수 있다.(`use_bias=False`)

```python
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Activation
from tensorflow.keras.models import Sequential

model = Sequential([
  Flatten(input_shape=[28, 28]),
  BatchNormalization(),
  Dense(300, kernel_initializer='he_normal', use_bias=False),
  BatchNormalization(),
  Activation('elu'),
  Dense(100, kernel_initializer='he_normal', use_bias=False),
  BatchNormalization(),
  Activation('elu'),
  Dense(10, activation='softmax')
])
```

- `BatchNormalization` 클래스는 조정할 하이퍼파라미터가 적다.
  - 가끔 `momentum` 매개변수를 변경이 필요(지수 이동 평균을 업데이트할 때 사용)
    - 새로운 값 `v`(입력 평균 벡터 or 표준편차 벡터)가 주어지면 다음 식을 사용해 이동 평균 $\hat v$를 업데이트한다.

    $$ \hat{v} \leftarrow \hat{v} \times momentum + v \times (1 - momentum)$$

    - 적절한 모멘텀 값은 일반적으로 1에 가깝다.(데이터셋이 크고 미니배치가 작으면 1에 더 가깝게 조정)
  
  - `axis` 하이퍼파라미터는 정규화할 축을 결정하며 기본값은 -1(마지막 축)이다.
    - 입력 배치가 2D이면 각 입력 특성이 배치에 있는 모든 샘플에 대해 계산한 평균과 표준편차를 기반으로 정규화된다.

## 11.1.4 그레이디언트 클리핑

그레이디언트 폭주 문제를 완화하는 방법으로 역전파될 때 일정 임계값을 넘어서지 못하게 그레이디언트를 잘래내는 **그레이디언트 클리핑**이 있다.
  - 순환신경망에서 배치정규화를 사용하지 못할 때 유용
  - 케라스에서 그레이디언트 클리핑을 구현하려면 옵티마이저를 만들 때 `clipvalue`와 `clipnorm`매개변수를 지정하면 된다.
    - `clipvalue` : 그레이디언트 벡터 방향을 바꿀 수 있다.
    - `clipnorm` : 그레이디언트 벡터 방향을 바꾸지 못한다. 

```python 
optimizer = tensorflow.keras.SGD(clipvalue=1.0) # 손실의 모든 편미분 값을 -1.0에서 1.0으로 잘라낸다.
model.compile(loss='mse', optimizer=optimizer)
```

# 11.2 사전훈련된 층 재사용하기

**전이 학습(Transfer Learning)**은 해결하려는 것과 비슷한 유형의 문제를 처리한 신경망이 이미 있는지 찾아본 다음, 그 신경망의 하위층을 재사용하는 방법을 말한다.
- 훈련 속도를 크게 높이고 필요한 훈련 데이터도 크게 줄여준다.

<img src="https://user-images.githubusercontent.com/78655692/148234268-c7aa8781-0f7e-401f-9baf-e188b9c3a182.png" height="500px" width = "500px">

> 만약 원래 문제에서 사용한 것과 크기가 다른 이미지를 입력으로 사용한다면 원본 모델에 맞는 크기로 변경하는 전처리 단계를 추가해야 한다. 일반적으로 전이 학습은 저수준 특성이 비슷한 입력에서 잘 작동한다.

전이 학습 시 보통 원본 모델의 출력층을 바꿔야 한다.
- 이 층이 새로운 작업에 가장 유용하지 않는 층이고 새로운 작업에 필요한 출력 개수와 맞지 않을 수도 있다.
- 원본 모델의 하위 은닉층이 훨씬 유용함

**전이학습 방법**<br>
- 재사용하는 층을 모두 동결(경사 하강법으로 가중치가 바뀌지 않도록 훈련되지 않는 가중치로 만듦)
- 모델을 훈련하고 성능을 평가
- 맨 위에 있는 한두개의 은닉층의 동결을 해제하고 역전파를 통해 가중치를 조정하여 성능이 향상되는지 확인
  - 훈련 데이터가 많을수록 많은 층의 동결 해제 가능
  - 재사용 층의 동결을 해제할 때는 학습률을 줄이는 것이 효율적이며 가중치를 세밀하게 튜닝하는 데 도움을 줌
- 성능이 좋아지지 않거나 훈련 데이터가 적을 경우 상위 은닉층 제거 후 남은 은닉층을 다시 동결하고 훈련
  - 훈련 데이터가 아주 많은 경우 은닉층을 제거하기보다는 다른 것으로 바꾸거나 더 많은 은닉층을 추가

## 11.2.1 케라스를 사용한 전이 학습

- 먼저 모델 A를 로드하고 이 모델의 층을 기반으로 새로운 모델을 만든다.
- 가정: model_A가 존재
  - 샌들과 셔츠를 제외한 8개의 클래스만 담겨 있는 패션 MNIST 활용하여 학습한 모델
- 문제점: 셔츠와 샌들을 분류하는 이진 분류기 model_B 훈련을 하려고 하지만 데이터가 매우 적음
- 목표: 전이학습을 이용한 model_B_on_A 훈련
  - 출력층만 제외하고 모든 층을 재사용

```python
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense

model_A = load_model("my_model_A.h5")
model_B_on_A = Sequential(model_A.layers[:-1])
model_B_on_A = (Dense(1, activation="sigmoid"))
```

위의 코드는 model_B_on_A를 훈련할 때 model_A도 영향을 받는다. 이를 원하지 않는 경우 model_A를 클론한다.
- `clone_model()` 메서드로 모델 A의 구조를 복제한 후 가중치를 복사(가중치는 별도로 복사가 필요)

```python
from tensorflow.keras.models import clone_model

model_A_clone = clone_model(model_A)
model_A_clone.set_weights(model_A.get_weights())
```

- 새로운 출력층이 랜덤하게 초기화되어 있으므로 큰 오차를 만들 것이다. 이를 피하기 위해서 처음 몇 번의 에포크 동안 재사용된 층을 동결하고 새로운 층에게 적절한 가중치를 학습할 시간을 준다.
   - 모든 층의 trainable 속성을 False로 지정하고 모델을 컴파일한다.
  
> 층을 동결하거나 동결을 해제한 후 반드시 모델을 컴파일해야 한다.

```python
from tensorflow.keras.optimizers import SGD

# 재사용 층 동결
for layer in model_B_on_A.layers[:-1]:
  layer.trainable = False

model_B_on_A.compile(loss="binary_crossentropy", optimizer="sgd",
                    metrics=["accuracy"])

# 큰 오차를 피하기 위한 몇 번의 에포크 동안 모델 훈련
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=4,
                          validation_data=(X_valid_B, y_valid_B))

# 재사용 층 동결해제
for layer in model_B_on_A.layers[:-1]:
  layer.trainable = True

optimizer = SGD(lr=1e-4)  # 기본 학습률은 1e-2

model_B_on_A.compile(loss="binary_crossentropy", optimizer=optimizer,
                    metrics=["accuracy"])
  
# 모델 학습
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=16,
                          validation_data=(X_valid_B, y_valid_B))

# 모델 평가
model_B_on_A.evaluate(X_test_B, y_test_B)
# [0.06887910133600235, 0.9925]
```

- 얕은 신경망 모델에서는 전이학습 성능이 좋지 않다.
- 전이 학습은 조금 더 일반적인 특성을 감지하는 경향이 있는 심층 합성곱 신경망에서 잘 동작한다.

## 11.2.2 비지도 사전훈련

레이블된 훈련 데이터가 많지 않은 복잡한 문제가 있는데, 이 작업에 대해 훈련된 모델을 핮을 수 없을 때 **비지도 사전훈련**을 수행하여 문제를 해결할 수 있다.
- 레이블되지 않은 훈련 데이터를 많이 모을 수 있는 경우에 오토인코더(autoencoder)나 생성적 적대 신경망과 같은 비지도 학습 모델을 훈련
  - 오토인코더나 GAN 판별자의 하위층을 재사용하고 그 위에 새로운 작업에 맞는 출력증을 추가한 후 지도 학습으로 최종 네트워크를 세밀하게 튜닝

## 11.2.3 보조 작업에서 사전훈련

레이블된 훈련 데이터가 많지 않는 경우에 또 다른 방법으로는 레이블된 훈련 데이터를 쉽게 얻거나 생성할 수 있는 보조 작업에서 첫 번째 신경망을 훈련하는 것이다.
- 이 신경망의 하위층을 실제 작업을 위해 재사용한다.
- 첫 번째 신경망의 하위층은 두 번째 신경망에 재사용될 수 있는 특성 추출기를 학습하게 된다.

> **자기 지도 학습**은 데이터에서 스스로 레이블을 생성하고 지도 학습 기법으로 레이블된 데이터셋에서 모델을 훈련하는 방법으로 사람이 레이블을 부여할 필요가 없어 비지도 학습의 형태로 분류된다.
