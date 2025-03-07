---
title: "10장 케라스를 사용한 인공 신경망 소개(3)"
excerpt: "하이퍼파라미터 튜닝"

categories:
  - 핸즈온 머신러닝
tags:
  - [hands-on]

permalink: /hands-on/ANN-3/

toc: true
toc_sticky: true
math: true

date: 2025-03-06
last_modified_at: 2025-03-07
---

# 10.3 신경망 하이퍼파라미터 튜닝하기

신경망에는 조정할 하이퍼파라미터가 많다. 최적의 하이퍼파라미터 조합을 찾는 방식에는 검증 세트에서 (또는 K-fold 교차 검증으로) 가장 좋은 점수를 내는지 확인하는 것이다. 
- `GridSearchCV`나 `RandomizedSearchCV`를 사용해 하이퍼파라미터 공간을 탐색할 수 있다.

```python
def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape)) # 입력 크기
    for layer in range(n_hidden):   # 은닉층 개수
        model.add(keras.layers.Dense(n_neurons, activation="relu")) # 뉴런 개수
    model.add(keras.layers.Dense(1))    # 출력층
    optimizer = keras.optimizers.SGD(lr=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model

keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model) # KerasRegressor 클래스의 객체 생성
```
- `KerasRegressor` 객체는 `build_model()` 함수로 만들어진 케라스 모델을 감싸는 간단한 래퍼이다.
    - 하이퍼파라미터를 지정하지 않았으므로 기본 하이퍼파라미터를 사용
- 사이킷런 회귀 추정기처럼 사용

```python
keras_reg.fit(X_train, y_train, epochs=100,
             validation_data=(X_valid, y_valid),
             callbacks=[keras.callbacks.EarlyStopping(patience=10)])
mse_test = keras_reg.score(X_test, y_test)
y_pred = keras_reg.predict(X_new) # 출력 점수는 음수의 MSE
```

- 하이퍼파라미터가 많으므로 랜덤 탐색을 사용하는 것이 좋다.

```python
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

param_distribs = {
    "n_hidden": [0, 1, 2, 3],
    "n_neurons": np.arange(1, 100),
    "learning_rate": reciprocal(3e-4, 3e-2),
}
rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3)
rnd_search_cv.fit(X_train, y_train, epochs=100,
                  validation_data=(X_valid, y_valid), # 조기 종료에만 사용
                  callbacks=[keras.callbacks.EarlyStopping(patience=10)])

# 최상 하이퍼파라미터와 훈련된 케라스 모델 출력
rnd_search_cv.best_params_
# {'learning_rate': 0.0033625641252688094, 'n_hidden': 2, 'n_neurons': 42}

rnd_search_cv.best_score_
# -0.3189529188278931

model = rnd_search_cv.best_estimator_.model
```

**하이퍼파라미터 최적화 라이브러리**<br>
- Hyperopt
- Hyperas, kopt, Talos
- 케라스 튜너
- Scikit-Optimize(skopt)
- Spearmint
- Hyperband
- Sklearn-Deap

## 10.3.1 은닉층 개수

복잡한 문제에서는 심층 신경망이 얕은 신경망보다 **파라미터 효율성**이 훨씬 좋다. 심층 신경망은 복잡한 함수를 모델링 하는 데 얕은 신경망보다 훨씬 적은 수의 뉴런을 사용하므로 동일한 양의 훈련 데이터에서 더 높은 성능을 낼 수 있다.

**전이 학습(transfer learning)**: 네트워크의 하위 층을 재사용하여 고수준 구조만 학습
- 하나 또는 두 개의 은닉층만으로도 많은 문제를 잘 해결할 수 있음

## 10.3.2 은닉층의 뉴런 개수

은닉층의 구성 방식은 일반적으로 각 층의 뉴런을 점점 줄여서 깔때기처럼 구성한다. 하지만 요즘엔 대부분 모든 은닉층에 같은 크기를 사용해도 동일하거나 더 나은 성능을 낸다. 또한 튜닝할 하이퍼파리미터가 층마다 한 개씩이 아니라 전체를 통틀어 한 개가 된다. 첫 번째 은닉층을 다른 은닉층보다 크게 하는 것이 도움이 된다.

**스트레치 팬츠 기법(효과적인 네트워크 설계)**<br>
- 필요보다 더 많은 층과 뉴런을 가진 모델을 선택한다.
- 과대적합되지 않도록 조기 종료나 규제 기법을 사용한다.
- 모델에서 문제를 일으키는 병목층을 피할 수 있다.
- 많은 뉴런을 가지기 때문에 입력에 있는 유용한 정보를 모두 유지할 수 있는 표현 능력을 갖는다.

## 10.3.3 학습률, 배치 크기 그리고 다른 하이퍼파라미터

다층 퍼셉트론에는 은닉층과 뉴런 개수 외에도 다른 하이퍼파라미터가 있다.

### 학습률

가장 중요한 하이퍼파라미터로 일반적인 최적의 학습률은 최대 학습률의 절반 정도이다. 좋은 학습률을 찾는 방법은 낮은 학습률에서 시작해 점진적으로 매우 큰 학습률까지 수백 번 반복하여 모델을 훈련하는 것이다.

- 반복마다 일정한 값을 학습률에 곱한다.
- 초반에는 학습률에 대한 손실이 줄어드지만 학습률이 커질수록 손실이 다시 커진다.
- 최적의 학습률은 손실이 다시 상승하는 지점보다 조금 아래의 값이다.

### 옵티마이저

고전적인 평범한 미니배치 경사 하강법보다 더 좋은 옵티마이저를 선택하고 이 옵티마이저의 하이퍼파라미터를 튜닝한다.

### 배치 크기

배치 크기는 모델 성능과 훈련 시간에 큰 영향을 미칠 수 있다. 큰 배치 크기를 사용하는 것의 주요 장점은 GPU와 같은 하드웨어 가속기를 효율적으로 활용할 수 있다. 즉, 훈련 알고리즘이 초당 더 많은 샘플을 처리할 수 있다.

작은 배치가 더 좋은 성능을 낼 수도 있으며 큰 배치를 사용해도 일반화 성능에 영향을 미치지 않는다는 논문도 존재한다.<br>
즉, 학습률 예열을 사용해 큰 배치 크기를 시도해보고 만약 훈련이 불안정하거나 최종 성능이 만족스럽지 못하면 작은 배치 크기를 사용하면 된다.

### 활성화 함수

일반적으로 ReLU 활성화 함수가 모든 은닉층에 좋은 기본값이지만 이외에도 다양한 활성화 함수가 존재한다. 출력층의 활성화 함수는 수행하는 작업에 따라 달라진다.

<img src="https://miro.medium.com/v2/resize:fit:1400/1*SgXNIupQ0EMXWItz74CLDw.png" height="400px" widht="500px">

### 반복 횟수

대부분의 경우 훈련 반복 횟수는 튜닝할 필요가 없으며 대신 조기 종료를 사용한다.








