---
title: "10장 케라스를 사용한 인공 신경망 소개(2)"
excerpt: "텐서플로 & 케라스 / 딥러닝 / 하이퍼파라미터 튜닝 "

categories:
  - 핸즈온 머신러닝
tags:
  - [hands-on]

permalink: /hands-on/ANN-2/

toc: true
toc_sticky: true
math: true

date: 2025-02-08
last_modified_at: 2025-02-09
---

# 10.2 케라스로 다층 퍼셉트론 구현하기

- 케라스는 모든 종류의 신경망을 손쉽게 만들고 훈련, 평가, 실행할 수 있는 고수준 딥러닝 API이다.
- 텐서플로와 케라스 다음으로 가장 인기 있는 딥러닝 라이브러리는 페이스북 파이토치 (PyTorch)이다.

## 10.2.1 텐서플로 2 설치

```
$ cd $ML_PATH   # ML을 위한 작업 디렉토리
$ source my_env/bin/activate  # 리눅스나 맥OS에서
$ .\my_env\Scripts\activate   # Windows에서

# 텐서플로 설치
$ python3 -m pip install -U tensorflow
```

- 텐서플로 설치 관련 문서: https://tensorflow.org/install

```python
# 텐서플로 & 케라스 설치 확인
import tensorflow as tf
from tensorflow import keras

tf.__version__
keras.__version__
```

## 10.2.2 시퀀셜 API를 사용하여 이미지 분류기 만들기

- 데이터: 28 x 28 크기의 패션 아이템 이미지
- 목표: 패션 아이템 분류

### 케라스를 사용하여 데이터셋 적재하기

```python
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

print(X_train_full.shape) # (60000, 28, 28)
print(X_train_full.dtype) # uint8

# 경사 하강법으로 신경망을 훈련하기 때문에 입력 스케일을 조정. 255(최대 범위)로 나눠서 0~1 사이로 조정
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.0

# 클래스 이름의 리스트 만들기
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag" ,"Ankle boot"]
```

### 시퀀셜 API를 사용하여 모델 만들기

아래의 모델은 2개의 은닉층으로 이루어진 다층 퍼셉트론이다.

```python
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Sequential

model = Sequential()    # 순서대로 연결된 층을 일렬로 쌓아서 구성한다
model.add(Flatten(input_shape=[28, 28]))    # 입력 이미지를 1D 배열로 변환
model.add(Dense(300, activation='relu'))   # 뉴런 300개를 가진 Dense 은닉층을 추가한다. 이 층은 ReLU 활성화 함수를 사용한다. Dense 층마다 각자 가중치 행렬을 관리한다. 이 행렬에는 층의 뉴런과 입력 사이의 모든 연결 가중치가 포함된다. 또한 (뉴런마다 하나씩 있는) 편향도 벡터로 관리한다. 
model.add(Dense(300, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))    # (클래스마다 하나씩) 뉴런 10개를 가진 Dense 출력층을 추가한다. 배타적인 클래스이므로 소프트맥스 활성화 함수를 사용한다.
```

위의 방식은 층을 하나씩 추가하는 방식이다. 하지만, Sequential 모델을 만들 때 층의 리스트를 전달하는 방식으로 모델을 만들 수도 있다.

```python
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
```
모델의 `summary()` 메서드는 모델에 있는 모든 층을 출력한다. 각 층의 이름, 출력 크기, 파라미터 개수가 함께 출력된다. 마지막에 훈련된는 파라미터와 훈련되지 않은 파라미터를 포함하여 전체 파라미터 개수를 출력한다.

```python
model.summary()

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# flatten (Flatten)            (None, 784)               0         
# _________________________________________________________________
# dense (Dense)                (None, 300)               235500    
# _________________________________________________________________
# dense_1 (Dense)              (None, 100)               30100     
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                1010      
# =================================================================
# Total params: 328,810
# Trainable params: 328,810
# Non-trainable params: 0
# _________________________________________________________________
```
- 첫 번째 은닉층은 784 x 300 개의 연결 가중치와 300개의 편향을 가진다. 이를 더하면 235,500개의 파라미터가 있다.
    - 이런 모델은 훈련 데이터를 학습하기 충분한 유연성을 갖는다.
    - 과대적합의 위험을 가지고 있다.
- 두 번째 은닉층은 300 x 100 개의 연결 가중치와 100개의 편향을을 갖는다.

모델에 있는 층의 리스트를 출력하거나 인덱스로 층을 쉽게 선택할 수 있다. 또한, 이름으로도 층을 선택할 수 있다.

```python
model.layers
# [<tensorflow.python.keras.layers.core.Flatten at 0x132414e48>,
#  <tensorflow.python.keras.layers.core.Dense at 0x1324149b0>,
#  <tensorflow.python.keras.layers.core.Dense at 0x1356ba8d0>,
#  <tensorflow.python.keras.layers.core.Dense at 0x13240d240>]

hidden1 = model.layers[1]
hidden1.name
# 'dense'

model.get_layer('dense') is hidden1
# True
```
층의 모든 파라미터는 `get_weights()` 메서드와 `set_weights()` 메서드를 사용해 접근할 수 있다. Dense 층의 경우 연결 가중치와 편향이 모두 포함되어 있다.
```python
weights, biases = hidden1.get_weights()
weights
# array([[ 0.02448617, -0.00877795, -0.02189048, ..., -0.02766046,
#          0.03859074, -0.06889391],
#        ...,
#       [-0.06022581, 0.01577859, -0.02585464, ..., -0.00527829,
#         0.00272203, -0.06793761]], dtype=float32)

weights.shape
# (784, 300)
biases
# array([0., 0., 0., 0., 0., 0., 0., 0., 0., ..., 0., 0., 0.], dtype=1oat32)

biases.shape
# (300, )
```

Dense 층은 대칭성을 깨뜨리기 위해 연결 가중치를 무작위로 초기화한다. 편향은 0으로 초기화한다. 다른 초기화 방법으로는 층을 만들 때 `kernel_initializer`와 `bias_initializer` 매개변수를 설정할 수 있다.

> 가중치 행렬의 크기는 입력의 크기에 달려 있다. 그래서 Sequential 모델에 첫 번째 층을 추가할 때 `input_shape` 매개변수를 지정한 것이다. 하지만, 입력 크기를 지정하지 않아도 상관은 없다. 케라스는 모델이 `build()` 메서드를 호출할 때 입력 크기를 받기 때문이다. 하지만, 지정을 하지않으면 `summary()` 메서드 호출이나 모델 저장 등 특정 작업을 수행하지 못하기 때문에 입력 크기를 알고 있다면 지정하는 것이 좋다.

### 모델 컴파일
모델을 만들고 나서 `compile()` 메서드를 호출하여 사용할 손실 함수와 옵티마이저(optimizer)를 지정해야 한다. 부가적으로 훈련과 평가 시에 계산할 지표를 추가로 지정할 수 있다.

```python
model.compile(loss="sparse_categorical_crossentropy",
            optimizer="sgd",
            metrics=["accuracy"])
```

- 클래스가 배타적이므로 `sparse_categorical_crossentropy` 손실을 사용한다.
    - 만약 샘플마다 클래스별 타깃 확률을 가지고 있다면 대신 `categorical_crossentropy` 손실을 사용해야 한다.
    - 이진 분류나 다중 레이블 이진 분류를 수행한다면 `sigmoid` 함수를 사용하고 `binary_Crossentropy` 손실을 사용한다.
- `sgd = stochastic gradient descent (default = lr=0.01)`
    - 옵티마이저에 `sgd`를 지정하면 기본 확률적 경사 하강법을 사용하여 모델을 훈련한다는 의미이며 다른 말로는 역전파 알고리즘을 수행하는 것이다.

### 모델 훈련과 평가
모델을 훈련하려면 `fit()` 메서드를 호출한다.

```python
history = model.fit(X_trian, y_train, epochs=30, validation_data=(X_valid, y_valid))

# Train on 55000 samples, validation on 5000 samples
# Epoch 1/30
# 55000/55000 [==============================] - 12s 6ms/step - loss: 1.0080 - accuracy: 0.6764 - val_loss: 0.5268 - val_accuracy: 0.8168
# Epoch 2/30
# 55000/55000 [==============================] - 10s 6ms/step - loss: 0.5108 - accuracy: 0.8239 - val_loss: 0.4558 - val_accuracy: 0.8476
# [...]
# Epoch 30/30
# 55000/55000 [==============================] - 10s 6ms/step - loss: 0.2277 - accuracy: 0.9185 - val_loss: 0.3012 - val_accuracy: 0.8912
```

- 입력 특성과 타깃 클래스, 훈련할 에포크 횟수를 전달한다. 검증 세트도 전달한다. (optional)
    - `X_train` : 입력 특성
    - `y_train` : 타깃 클래스
    - `epochs` : 훈련할 에포크 횟수
- 케라스는 에포크가 끝날 때마다 검증 세트를 사용해 손실과 추가적인 측정 지표를 계산한다.
    - 이 지표는 모델이 얼마나 잘 수행되는지 확인하는 데 유용하다.
    - 훈련 세트 성능이 검증 세트보다 월등히 높다면 과대적합되었을 것이다.
- 케라스는 (진행 표시줄과 함께) 처리한 샘플 개수와 샘플마다 걸린 평균 훈련 시간, 훈련 세트와 검증 세트에 대한 손실과 정확도를 출력한다.

**데이터의 클래스가 불균형하다면?**<br>
- 어떤 클래스는 많이 등장하고 다른 클래스는 조금 등장하여 훈련 세트가 편중되어 있다면 `fit()` 메서드를 호출할 때 `class_weight` 매개변수를 지정하는 것이 좋다.
    - 적게 등장하는 클래스는 높은 가중치를 부여하고 많이 등장하는 클래스는 낮은 가중치를 부여한다.
    - 케라스가 손실을 계산할 때 이 가중치를 사용한다.
- 샘플별로 가중치를 부여하고 싶다면 `sample_weight` 매개변수를 지정한다.

`fit()` 메서드가 반환하는 `History` 객체에는 훈련 파라미터(history.params), 수행된 에포크 리스트(history.epoch)가 포함된다. 이 객체에서 가장 중요한 속성은 에포크가 끝날 때마다 훈련 세트와 검증 세트에 대한 손실과 측정한 지표를 담은 딕셔너리(history.history)이다. 이를 통해 모델의 학습 곡선을 볼 수 있다.

```python
import pandas as pd
import matplotlib.pyplot as plt

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)   # 수직축의 범위를 [0-1] 사이로 설정
plt.show()
```
<img src="https://user-images.githubusercontent.com/78655692/146779584-7d03fd28-b752-4826-bfb4-b6a26c2bb9e2.png">

- 훈련하는 동안 훈련 정확도와 검증 정확도가 꾸준히 상승하고 훈련 손실과 검증 손실은 감소한다.
- 검증 손실은 에포크가 끝난 후에 계산되고 훈련 손실은 에포크가 진행되는 동안 계산된다.
- 케라스에서는 `fit()` 메서드를 다시 호출하면 중지되었던 곳에서부터 훈련을 이어갈 수 있다.

**만약 모델 성능 좋지 않다면?**<br>
모델 성능에 만족스럽지 않으면 처음부터 되돌아가서 하이퍼파라미터를 튜닝해야 한다. 그래도, 여전히 성능이 높지 않으면 층 개수, 층에 있는 뉴런 개수, 은닉층이 사용하는 활성화 함수와 같은 모델의 하이퍼파리미터를 튜닝해 본다.

모델의 검증 정확도가 만족스러운 경우, 배포 전 테스트 세트로 모델을 평가하여 일반화 오차를 추정해야 한다. 이때는 `evaluate()` 메서드를 사용한다.

```python
model.evaluate(X_test, y_test)
# 10000/10000 [==========] - 0s 29us/sample - loss: 0.3340 - acci」racy: 0.8851
# [0.3339798209667206, 0.8851]
```

### 모델을 사용해 예측을 만들기
모델의 `predict()` 메서드를 사용해 새로운 샘플에 대해 예측을 만들 수 있다.
- `predict()`: 각 클래스에 속할 확률 계산
- `predict_classes`: 가장 높은 확률의 클래스 지정

```python
X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)
# array([[0. , 0. , 0.  , 0. , 0.  , 0.03, 0. , 0.01, 0.  , 0.96],
#        [0. , 0. , 0.98, 0. , 0.02, 0.  , 0. , 0.  , 0.  , 0. ],
#        [0. , 1. , 0.  , 0. , 0.  , 0.  , 0. , 0.  , 0.  , 0.]],
#       dtype=float32)

y_pred = model.predict_classes(X_new)
y_pred
# array([9, 2, 1])

np.array(class_names)[y_pred]
# array(['Ankle boot', 'Pullover', 'Trouser'], dtype='<U11')
```

## 10.2.3 시퀀셜 API를 사용하여 회귀용 다층 퍼셉트론 만들기

## 10.2.4 함수형 API를 사용해 복잡한 모델 만들기

## 10.2.5 서브클래싱 API로 동적 모델 만들기

## 10.2.6 모델 저장과 복원

## 10.2.7 콜백 사용하기

## 10.2.8 텐서보를 사용해 시각화하기


# 10.3 신경망 하이퍼파라미터 튜닝하기

