---
title: "10장 케라스를 사용한 인공 신경망 소개(2)"
excerpt: "회귀 & 분류 / 시퀀셜 API / 함수형 API / 서브클래싱 API"

categories:
  - 핸즈온 머신러닝
tags:
  - [hands-on]

permalink: /hands-on/ANN-2/

toc: true
toc_sticky: true
math: true

date: 2025-02-16
last_modified_at: 2025-03-06
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

- 데이터셋: 캘리포니아 주택 가격
- 목표: 주택 가격 예측

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_test_full)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)
```

시퀀셜 API를 사용해 회귀용 MLP를 구축, 훈련, 평가, 예측하는 방법이 분류와 비슷하지만 회귀에서는 출력층의 활성화 함수가 없고 손실 함수로 평균 제곱 오차를 사용한다. 

```python
# 데이터셋에 잡음이 많아 과대적합이 될 수 있음 -> 뉴런 수가 적은 은닉층 사용
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    keras.layers.Dense(1)
])
model.compile(loss="mean_squared_error", optimizer="sgd")
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))
mse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3]  # 새로운 샘플로 가정
y_pred = model.predict(X_new)
```

**케라스 Sequential 클래스 장단점**<br>

- 사용하기 매우 쉬우며 성능 우수하다.
- 입출력이 여러 개이거나 더 복잡한 네트워크를 구성하기 어려움<br>
해결책: Sequential 클래스 대신에 함수형 API, 하위클래스(subclassing) API 등을 사용하여 보다 복잡하며, 보다 강력한 딥러닝 모델을 구축 가능하다.

## 10.2.4 함수형 API를 사용해 복잡한 모델 만들기

- 일반적인 MLP
    - 네트워크에 있는 층 전체에 모든 데이터를 통과
    - 데이터에 있는 간단한 패턴이 연속된 변환으로 인해 왜곡될 수 있다.

- Wide & Deep 신경망 
    - 입력의 일부 또는 전체가 출력층에 바로 연결하는 순차적이지 않은 신경망
    - 신경망이 복잡한 패턴과 간단한 규칙을 모두 학습할 수 있다. 

<img src="https://user-images.githubusercontent.com/78655692/147923150-82adc84b-c493-4c66-99c8-902a8250b41c.png" height="500px" width="400px">

```python
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model

input_ = keras.layers.Input(shape=X_train.shape[1:]) # shape와 dtype을 포함하여 모델의 입력을 정의
hidden1 = keras.layers.Dense(30, activation="relu")(input_) # 30개의 뉴런과 ReLU 활성화 함수를 가진 Dense 층을 만든다. 이 층은 만들어지자마자 입력과 함께 함수처럼 호출
# 이 메서드에는 build() 메서드를 호출하여 층의 가중치를 생성
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1) # 첫 번쨰 층의 출력을 전달
concat = keras.layers.Concatenate()([input_, hidden2])     # Concatenate 층을 만들고 또 다시 함수처럼 호출하여 두 번째 은닉층의 출력과 입력을 연결
output = keras.layers.Dense(1)(concat) # 하나의 뉴런과 활성화 함수가 없는 출력층을 만들고 Concatenate 층이 만든 결과를 사용해 호출
model = keras.Model(inputs=[input_], outputs=[output]) # 사용할 입력과 출력을 지정하여 케라스 Model을 만듦
```
나머지는 이전과 동일하게 모델을 컴파일한 다음 훈련, 평가, 예측을 수행할 수 있다.

<img src="https://user-images.githubusercontent.com/78655692/147924063-21fd1cc2-0e97-4ec2-aad4-dfe0f2fb27d3.png" height="500px" width="400px">

```python
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.models import Model

input_A = keras.layers.Input(shape=[5], name="wide_input")
input_B = keras.layers.Input(shape=[6], name="deep_input")
hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input_A, hidden2])
output = keras.layers.Dense(1, name="output")(concat)
model = keras.Model(inputs=[input_A, input_B], outputs=[output])
```

모델 컴파일은 이전과 동일하지만 fit() 메서드를 호출할 때 하나의 입력 행렬 X_train을 전달하는 것이 아니라 입력마다 하나씩 행렬의 튜플 (X_train_A, X_train_B)을 전달해야 한다. X_valid에도 동일하게 적용된다.

```python
from tensorflow.keras.optimizers import SGD

model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))

X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:] # input_A 입력값 : 0~4번 인덱스 특성, input_B 입력값 : 2~7번 인덱스 특성
X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:]
X_test_A, X_test_B = X_test[:, 5:], X_test[:, 2:]
X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]

history = model.fit((X_train_A, X_train_B), y_train, epochs=20,
                    validation_data=((X_valid_A, X_valid_B), y_valid))
mse_test = model.evaluate((X_test_A, X_test_B), y_test)
y_pred = model.predict((X_new_A, X_new_B))
```

여러 개의 출력이 필요한 경우는 다음과 같다.

- 다중 출력: 회귀 작업과 분류 작업을 함께 하는 경우
    - 그림에 있는 주요 물체를 분류하고 위치를 알아야 하는 경우

- 동일한 데이터에서 독립적인 여러 작업을 수행하는 경우
    - 얼굴 사진으로 다중 작업 분류: 얼굴 표정 분류 & 안경의 유무 분류

- 규제 기법 사용: 과대적합 감소와 모델의 일반화 성능 높이도록 훈련에 제약
    - 신경망 구조 안에 보조 출력을 추가하여 이를 사용해 하위 네트워크가 나머지 네트워크에 의존하지 않고 그 자체로 유용한 것을 학습하는 지 확인할 수 있음

아래 그림은 규제 기법을 사용한 여러 개의 출력 신경망 그림이다.

<img src="https://user-images.githubusercontent.com/78655692/147926849-7e148d99-37b7-4642-b49a-1d288def3ee9.png" height="500px" width="400px">

```python
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.models import Model

input_A = keras.layers.Input(shape=[5], name="wide_input")
input_B = keras.layers.Input(shape=[6], name="deep_input")
hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input_A, hidden2])
output = keras.layers.Dense(1, name="main_output")(concat)
aux_output = keras.layers.Dense(1, name="aux_output")(hidden2)
model = keras.Model(inputs=[input_A, input_B], outputs=[output, aux_output])
```

- 각 출력은 자신만의 손실 함수가 필요하므로 모델을 컴파일할 때 손실의 리스트를 전달해야 한다.
- 케라스는 나열된 손실을 모두 더하여 최종 손실을 구해 훈련에 사용한다.
- 출력 별로 손실가중치를 지정하여 출력별 중요도 지정 가능(주 출력에 더 많은 가중치를 부여)

```python
model.compile(loss=['mse', 'mse'], loss_weight=[0.9, 0.1], optimizer="sgd")
```

- 모델을 훈련할 때 각 출력에 대한 레이블을 제공해야 한다.
- 여기에서는 주 출력과 보조 출력이 같은 것을 예측해야 하므로 동일한 레이블을 사용한다.

```python
history = model.fit(
    [X_train_A, X_train_B], [y_train_A, y_train_B], epochs=20,
    validation_data=([X_valid_A, X_valid_B], [y_valid, y_valid]))
```

모델을 평가할 떄, 케라스가 개별 손실과 함께 총 손실을 반환한다.
```python
total_loss, main_loss, aux_loss = model.evaluate(
    [X_test_A, X_test_B], [y_test, y_test])
```

`predict()` 메서드는 각 출력에 대한 예측을 반환한다.
```python
y_pred_main, y_pred_aux = model.predict([X_new_A, X_new_B])
```

## 10.2.5 서브클래싱 API로 동적 모델 만들기

- 시퀀셜 API와 함수형 API는 모두 선언적이다.(=정적)
    - 사용할 층과 연결 방식을 먼저 정의하고 모델에 데이터를 주입하여 훈련이나 추론을 시작할 수 있다.
    - 장점
        1. 모델을 저장하거나 복사, 공유하기 쉽다.
        2. 모델의 구조를 출력하거나 분석하기 좋다.
        3. 프레임워크가 크기를 짐작하고 타입을 확인하여 에러를 일찍 발견할 수 있다.
        4. 전체 모델이 층으로 구성된 정적 그래프이므로 디버깅하기 쉽다.

하지만 어떤 모델은 반복문을 포함하고 다양한 크기를 다루어야 하며 조건문을 가지는 등 여러 가지 동적인 구조를 필요로 한다.

**서브클래싱 API를 사용!!**

- `Model` 클래스를 상속한 다음 생성자 안에서 필요한 층을 만든다.
- 초기설정 메서드 `__init__()`를 이용하여 은닉층과 출력층 설정한다.
- `call()` 메서드 안에 수행하려는 연산을 기술한다.
- 이전과 동일하게 모델 컴파일, 훈련, 평가, 예측을 수행할 수 있다.

```python
class WideAndDeepModel(keras.Model):
    def __init__(self, units=30, activation='relu', **kwargs):
        super().__init__(**kwargs)  # 표준 매개변수를 처리(ex.name)
        self.hidden1 = keras.layers.Dense(units, activation=activation)
        self.hidden2 = keras.layers.Dense(units, activation=activation)
        self.main_output = keras.layers.Dense(1)
        self.aux_output = keras.layers.Dense(1)
    def call(self, inputs):
        input_A, input_B = inputs
        hidden1 = self.hidden1(input_B)
        hidden2 = self.hidden2(hidden1)
        concat = keras.layers.concatenate([input_A, hidden2])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return main_output, aux_output

model = WideAndDeepModel()
```
**장점**<br>
- 함수형 API와 매우 비슷하지만 Input 클래스의 객체를 만들 필요가 없는 대신 `call()` 메서드의 input 매개변수를 사용한다.
- 생성자(`__init__`)에 있는 층 구성과 `call()` 메서드에 있는 정방향 계산을 분리한다.
- `call()`메서드 안에서는 for문, if문, 텐서플로 저수준 연산 등 원하는 어떤 계산도 사용할 수 있다.

**단점**<br>
- 모델 구조가 call() 메서드 안에 숨겨져 있어 케라스가 쉽게 분석할 수 없으며 모델을 저장하거나 복사할 수 없다.
- `summary()` 메서드를 호출하면 층의 목록만 나열되고 층 간의 연결 정보를 얻을 수 없다.
- 케라스가 타입과 크기를 미리 확인할 수 없어 실수가 발생하기 쉽다.

> 높은 유연성이 필요하지 않는다면 시퀀스 API와 함수형 API를 사용하고 높은 유연성이 필요한 경우 서브클래싱 API를 사용하면 된다.


## 10.2.6 모델 저장과 복원

시퀀셜 API와 함수형 API를 사용하면 훈련된 케라스 모델을 쉽게 저장할 수 있다.

```python
# Model save
model = keras.models.Sequential([...])
model.compile([...])
model.fit([...])
model.save("my_keras_model.h5")

# Model load
model = keras.models.load_model("my_keras_model.h5")
```
- 모든 하이퍼파라미터를 포함하여 모델 구조, 층의 모든 모델 파라미터(연결 가중치 & 편향)와 옵티마이저를 저장한다.

> 서브클래싱 API에서는 사용할 수 없다. `save_weights()`와 `load_weights()` 메서드를 사용하여 모델 파라미터를 저장하고 복원할 수 있으며 이를 제외하면 모두 수동으로 저장하고 복원해야 한다.

## 10.2.7 콜백 사용하기

대규모 데이터셋을 훈련할 경우, 컴퓨터에 문제가 생겨 모든 것을 잃지 않으려면 훈련 마지막에 모델을 저장하는 것뿐만 아니라 훈련 도중 일정 간격으로 체크포인트를 저장해야 한다. 이때는 **콜백(callback)**을 사용하면 된다.

`fit()` 메서드의 `callbacks` 매개변수를 사용하여 케라스가 훈련의 시작이나 끝에 호출할 객체 리스트를 지정할 수 있다. 또는 에포크의 시작이나 끝, 각 배치 처리 전후에 호출할 수 있다.

- `ModelCheckpoint`는 훈련하는 동안 일정한 간격으로 모델의 체크포인트를 저장하며 매 에포크의 끝에서 호출한다.

```python
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

[...] # 모델을 만들고 컴파일하기
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5")
history = model.fit(X_train, y_train, epochs=10, callbacks=[checkpoint_cb])
```
- 훈련하는 동안 검증 세트를 사용하면 `ModelCheckpoint`를 만들 때 `save_best_only=True`로 지정하면 최상의 검증 세트 점수에서만 모델을 저장한다.

```python
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5",
                                                save_best_only=True)
history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb])
model = keras.models.load_model("my_keras_model.h5") # 최상의 모델 load
```

조기 종료를 구현하는 또 다른 방법은 `EarlyStopping` 콜백을 사용하는 것이다.<br>
- EarlyStopping 콜백은 일정 에포크(`patience` 매개변수로 지정) 동안 검증 세트에 대한 점수가 향상되지 않으면 훈련을 멈춘다.
- 모델이 향상되지 않으면 훈련이 자동으로 중지되기 때문에 에포크의 숫자를 크게 지정해도 된다.
- `restore_best_weights=True` : 최상의 모델 복원 기능 설정
    - 훈련이 끝난 후 최적 가중치 바로 복원

```python
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
                                                restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb, early_stopping_cb])
```

더 많은 제어를 원한다면 사용자 정의 콜백을 만들 수 있다.
- 훈련하는 동안 검증 손실과 훈련 손실의 비율을 출력(과대적합 감지)
```python
class PrintValTrainRatioCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print("\nval/train: {:.2f}".format(logs["val_loss"] / logs["loss"]))
```

## 10.2.8 텐서보드를 사용해 시각화하기

텐서보드는 인터렉티브 시각화 도구이다. 훈련하는 동안 학습 곡선을 그리거나 여러 실행 간의 학습 곡선을 비교하고 계산 그래프 시각화와 훈련 통계 분석을 수행할 수 있다. 또한 모델이 생성한 이미지를 확인하거나 3D에 투영된 복잡한 다차원 데이터를 시각화하고 자동으로 클러스터링을 해주는 등 많은 기능을 제공한다.

