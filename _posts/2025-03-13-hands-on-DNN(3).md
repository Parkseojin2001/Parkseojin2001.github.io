---
title: "11장 케라스를 사용한 인공 신경망 소개(3)"
excerpt: "과대적합/규제/드롭아웃"

categories:
  - 핸즈온 머신러닝
tags:
  - [hands-on]

permalink: /hands-on/DNN-3/

toc: true
toc_sticky: true
math: true

date: 2025-03-13
last_modified_at: 2025-03-13
---

# 11.4 규제를 사용해 과대적합 피하기

심층 신경망의 높은 자유도는 네트워크를 훈련 세트에 과대적합되기 쉽게 만들기 때문에 규제가 필요하다.
- 조기종료 기법: `EarlyStopping` 콜백을 사용하여 일정 에포크 동안 성능이 향상되지 않는 경우 자동 종료시키기
- 배치 정규화: 불안정한 그레이디언트 문제해결을 위해 사용하지만 규제용으로도 활용 가능(가중치 변화를 조절하는 역할)

## 11.4.1 $\ l_1$ 과 $\ l_2$ 규제

신경망의 연결 가중치를 제한하기 위해 $l_2$ 규제를 사용하거나 (많은 가중치가 0인) 희소 모델을 만들기 위해 $l_1$ 규제를 사용할 수 있다.
- `kernel_regularizer` 옵션을 이용해서 $l_2$ 규제를 적용

    ```python
    layer = keras.layers.Dense(100, activation='elu',
                            kernel_initializer="he_normal",
                            kernel_regularizer=keras.regularizers.l2(0.01))
    ```

- $l_1$ 규제가 필요한 경우 : kernel_regularizers.l1()을 이용
- $l_1$과 $l_2$ 규제가 필요한 경우: kernel_regularizers.l1_l2()를 사용

일반적으로 네트워크의 모든 은닉층에 동일한 활성화 함수, 동일한 초기화 전량을 사용하거나 모든 층에 동일한 규제를 적용하기 때문에 동일한 매개변수 값을 반복한다. 이는 코드를 읽기 어렵게 만들고 버그를 만들기 쉽게 한다.

**해결책**<br>

- 코드를 리팩터링(refactoring)을 이용
- 파이썬의 `functools.partial()` 함수를 사용하여 기본 매개변수 값을 사용하여 함수 호출을 감싼다.

```python
from functools import partial
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten

RegularizedDense = partial(Dense,
                           activation="elu",
                           kernel_initializer="he_normal",
                           kernel_regularizer=regularizers.l2(0.01))

model = Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    RegularizedDense(300),
    RegularizedDense(100),
    RegularizedDense(10, activation="softmax",
                    kernel_initializer="glorot_uniform")
])
```

## 11.4.2 드롭아웃

드롭아웃(Dense)은 심층 신경망에서 가장 인기 있는 규제 기법 중 하나이다.

<img src="https://user-images.githubusercontent.com/78655692/148241979-69c54fd9-dc5e-4736-9e98-701fd9f35cb6.png">

- 매 훈련 스텝에서 각 뉴런은 임시적으로 드롭아웃될 확률 $p$를 가진다. 즉, 이번 훈련 스텝에는 완전히 무시되지만 다음 스텝에는 활성화될 수 있다.
    - 하이퍼파라미터 $p$를 **드롭아웃 비율**이라고 하고 보통 10%와 50% 사이를 지정한다.
        - 순환 신경망: 20% ~ 30%
        - 합성곱 신경망: 40% ~ 50%
        - 순환 신경망: 20% ~ 30%
- 훈련이 끝난 후에는 뉴런에 더는 드롭아웃을 적용하지 않는다.
- 케라스에서는 `keras.layers.Dropout` 층을 사용하여 드롭아웃을 구현한다.



- 드롭아웃으로 훈련된 뉴런은 이웃한 뉴런에 맞추어 적응 될 수 었어 가능한 자기 자신이 유용해져야 한다.
- 몇 개의 입력 뉴런에만 지나치게 의존할 수 없으며 모든 입력 뉴런에 주의를 기울어야한다.
- 입력값의 작은 변화에 덜 민감해져 더 안정적인 네트워크가 되어 일반화 성능이 좋아진다.
- 각 훈련 스텝에서 고유한 네트워크가 생성된다.

> 일반적으로 (출력층을 제외한) 맨 위의 층부터 세 번째 층까지 있는 뉴런에만 드롭아웃을 적용한다.

드롭아웃을 적용할 때는 훈련이 끝난 뒤 각 입력의 연결 가중치에 **보존 확률** $(1-p)$을 곱해야 한다. 또는 훈련하는 동안 각 뉴런의 출력을 보존 확률로 나눌 수도 있다.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dropout, Dense
model = Sequential([
    Flatten(input_shape=[28, 28]),
    Dropout(rate=0.2),
    Dense(300, activation="elu", kernel_initializer="he_normal"),
    Dropout(rate=0.2),
    Dense(100, activation="elu", kernel_initializer="he_normal"),
    Dropout(rate=0.2),
    Dense(10, activation="softmax")
])
```
> - 드롭아웃은 훈련하는 동안에만 활성화되므로 훈련 손실과 검증 손실을 비교하면 오해를 일으키기 쉽다. 특히 비슷한 훈련 손실과 검중 손실을 얻었더라도 모델이 훈련 세트에 과대적합될 수 있다. 
- 훈련 후 드롭아웃을 끄고 훈련 손실을 재평가해야 한다.

- 모델이 과대적합되었다면 드롭아웃 비율을 늘리고 과소적합되었다면 드롭아웃 비율을 낮추어야 한다. 
- 층이 클 때는 드롭아웃 비율을 늘리고 작은 층에는 드롭아웃 비율을 낮추는 것이 도움이 된다.
- 많은 최신 신경망 구조는 마지막 은닉층 뒤에만 드롭아웃을 사용한다.

### 알파 드롭아웃

- SELU 활성화 함수를 기반으로 자기 정규화 네트워크를 규제할 때는 알파(alpha) 드롭아웃을 사용하는 것을 추천
    - 입력과 평균의 표준편차를 유지
    - 일반 드롭아웃은 자기 정규화 기능 방해

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, AlphaDropout, Dense

model = Sequential([
    Flatten(input_shape=[28, 28]),
    AlphaDropout(rate=0.2),
    Dense(300, activation="selu", kernel_initializer="lecun_normal"),
    AlphaDropout(rate=0.2),
    Dense(100, activation="selu", kernel_initializer="lecun_normal"),
    AlphaDropout(rate=0.2),
    Dense(10, activation="softmax")
])
```

## 11.4.3 몬테 카를로 드롭아웃

- 훈련된 드롭아웃 모델을 재훈련하거나 수정하지 않고 성능을 크게 향상시키는 기법
- 훈련된 모델의 예측기능을 이용한 결과를 스택으로 쌓은 후 평균값을 계산

```python
import numpy as np

y_pobas = np.stack([model(X_test_scaled, training=True)
                    for sample in range(100)])
y_proba = y_probas.mean(axis=0)
y_std = y_probas.std(axis=0)
```

> 몬테 칼를로 샘플의 숫자는 튜닝할 수 있는 하이퍼파라미터이다. 이 값이 높을수록 예측과 불확실성 추정이 더 정확해질 것이다. 하지만 샘플 수를 두배로 늘리면 예측 시간도 두 배가 된다. 또한 일정 샘플 수가 넘어서면 성능이 크게 향상되지 않는다. 즉, 성능과 정확도 사이에 균현점을 찾는 것이 중요하다.

모델이 훈련하는 동안 다르게 작동하는 층을 가지고 있다면 훈련 모들를 강제로 설정하면 안된다.
- Dropout 층을 MCDropout 클래스로 바꿔준다.

```python
from tensorflow.keras.layers import Dropout

class MCDropout(Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)
```

- `Dropout` 층을 상속하고 `call()` 메서드를 오버라이드하여 training 매개변수를 True로 강제로 설정한다.

## 11.4.4 맥스-노름 규제

맥스-노름 규제는 각각의 뉴런에 대해 입력의 연결 가중치 $w$가 $|w|_2 \leq r$이 되도록 제한한다.
- $r$ : 맥스-노름 하이퍼파라미터
    - $r$을 줄이면 규제의 양이 증가하여 과대적합을 감소시키는 데 도움이 된다.
- $\left\vert \cdot \right\vert \ _2$ : $l_2$ 노름
- 훈련 스텝이 끝나고 $\|w\|_2$ 를 계산하고 필요하면 $w$의 스케일을 조정한다.($w \leftarrow \frac{r}{\|w\|_2}$)

맥스-노름 규제는 (배치 정규화를 사용하지 않았을 때) 불안정한 그레이디언트 문제를 완화하는 데 도움을 줄 수 있다.

- 케라스에서 맥스-노름 규제를 구현하려면 적절한 최댓값으로 지정한 `max_norm()`이 반환한 객체로 은닉층의 `kernel_constraint` 매개변수를 지정한다.

```python
from tensorflow.keras.layers import Dense
from tensorflow.keras.constraints import max_norm

Dense(100, activaiton='elu', kernel_initializer='he_normal',
     kernel_constraint=max_norm(1,))
```
- 매 훈련 반복이 끝난 후 모델의 `fit()` 메서드가 층의 가중치와 함께 `max_norm()`이 반환한 객체를 호출하고 스케일이 조정된 가중치를 반환받는다. 
    - `kernel_constraint` : 사용자 정의 규제 함수 정의
    - `bias_constraint` : 편향을 규제
- `max_norm()`는 기본값이 0인 axis 매개변수가 있으며 이를 이용해 각 뉴런의 가중치 벡터에 독립적으로 적용된다.

# 11.5 요약 및 실용적인 가이드라인

**기본 DNN 설정**

|하이퍼파라미터|기본값|
|----------|----|
|커널 초기화|He 초기화|
|활성화 함수|ELU|
|정규화|얕은 신경망일 경우 없음. 깊은 신경망이라면 배치 정규화|
|규제|조기 종료(필요하면 $l_2 규제 추가)|
|옵티마이저|모멘텀 최적화(또는 RMSProp이나 Nadam)|
|학습률 스케줄|1사이클|

- 네트워크가 완전 연결 층을 쌓은 단순한 모델일 때는 자기 정규화 사용

**자기 정규화를 위한 DNN 설정**

|하이퍼파라미터|기본값|
|----------|----|
|커널 초기화|르쿤 초기화|
|활성화 함수|SELU|
|정규화|없음(자기 정규화)|
|규제|필요하다면 알파 드롭아웃|
|옵티마이저|모멘텀 최적화(또는 RMSProp이나 Nadam)|
|학습률 스케줄|1사이클|

**예외적인 경우**<br>
- 희소 모델이 필요한 경우
    - $l_1$ 규제를 사용할 수 있다.
    - 매우 희소한 모델이 필요하면 텐서플로 모델 최적화 툴킷을 사용할 수 있다.
    - 자기 정규화를 깨뜨리므로 이 경우 기본 DNN 설정을 사용해야 한다.
- 빠른 응답을 하는 모델이 필요한 경우
    - 층 개수를 줄이고 배치 정규화 층을 이전 층에 합친다.
    - LeakyReLU나 ReLU와 같이 빠른 활성화 함수를 사용한다.
    - 부동소수점 정밀도를 32비트에서 15비트 혹은 8비트로 낮출 수도 있다.
- 위험에 민감하고 예측 속도가 매우 중요하지 않은 경우
    - 불확실성 추정과 신뢰할 수 있는 확률 추정을 얻기 위해 MC 드롭아웃을 사용할 수 있다.