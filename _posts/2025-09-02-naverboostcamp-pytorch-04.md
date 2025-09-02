---
title: "선형 회귀"
description: "네이버 부스트캠프 PyTorch 강의 정리 포스트입니다."

categories: [Naver-Boostcamp, PyTorch]
tags: [Naver-Boostcamp, pytorch, linear-regression]

permalink: /naver-boostcamp/PyTorch/pytorch-04/

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-09-02
last_modified_at: 2025-09-02
---

선형 회귀란 주어진 트레이닝 데이터를 사용하여 특징 변수와 목표 변수 사이의 선형 관계를 분석하고, 이를 통해 모델을 학습시켜, 트레이닝 데이터에 포함되지 않은 새로운 데이터의 결과를 연속적인 숫자 값으로 예측하는 과정을 말한다.

대표적인 예로 임금 예측, 부동산 가격 예측이 있다.

### 트레이닝 데이터

만약 YearsExperience(연차)에 따른 임금(Salary) 데이터셋이 있을 때, 특징 변수는 YearsExperience이고 목표 변수는 Salary이다.

트레이닝 데이터를 불러오고 특징 변수와 목표 변수를 분리하는 코드는 아래와 같다.

```python
import pandas as pd

# 구분자는 ',' 이며 0번째 행을 column 명으로 사용
data = pd.read_csv('Salary_dataset.csv', sep=',', header = 0)   

x = data.iloc[:, 1].values
t = data.iloc[:, 2].values
```

### 상관 관계 분석

특징 변수와 목표 변수 간의 상관 관계를 분석하면 다음과 같은 정보를 얻을 수 있다.

- 두 변수 간의 선형 관계를 파악
- 그 관계가 양의 관계인지 또는 음의 관계인지 파악
- 높은 상관 관계를 가지는 특징 변수들을 파악(다중 선형 회귀 모델에서 필요)
    - 특징 변수가 너무 많은 경우, 모델 구현에 필요하지 않은 변수를 제거하여 모델의 복잡성을 줄이고 성능을 높여줄 수 있다.

상관 관계 분석을 할 때 사용하는 수식은 `표본상관계수`이다.

$$
r_{xt} = \frac{\sum_{i=1}^n (x_i - \bar{x})(t_i - \bar{t})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2\sum_{i=1}^n (t_i - \bar{t})^2}}
$$

표본상관계수는 두 변수가 함께 변화하는 정도를 각각 변화하는 정도로 나누어 두 변수 간의 관계를 파악한다.

이를 코드로 표현하면 `np.corrcoef(x, t)` 이다.

또한 코드의 결과값은 아래의 표를 참고하여 두 변수간의 상관관계의 정도를 파악하면 된다.

<img src="https://t1.daumcdn.net/cfile/tistory/99C148495C6AA1AA16">

이를 산점도로 시각화하면 아래와 같으며 산점도 그래프를 얻기 위해서는 `plt.scatter(x, t)`를 사용하면 된다.

<img src="https://wikidocs.net/images/page/253020/%EC%83%81%EA%B4%80%EA%B3%84%EC%88%98.png" width="500" height="400">

> 선형 회귀 모델 학습을 할 때는 데이터를 2치원 형태로 변환해야하며 이 변환은 `view(-1, 1)`을 통해 변환할 수 있다.

### 선형 회귀 모델 학습

선형 회귀 모델에서 **학습이란 주어진 트레이닝 데이터의 특성을 가장 잘 표현할 수 있는 직선 $y = wx + b$의 기울기(가중치) $w$와 $y$절편(바이어스) $b$를 찾는 과정**을 말한다.

신경망 관점에서 선형 회귀 모델은 입력층의 특징 변수가 출력층의 예측 변수로 mapping되는 과정이라고 말할 수 있다.

- 입력층
    - 특징 변수들을 포함하며, 각 특징 변수는 하나의 뉴런에 대응한다.
    - 각 뉴런의 **가중치와 바이어스**를 통해 출력층의 뉴런과 연결되며 이 떄 가중치와 바이어스를 파라미터라고 한다.
- 출력층
    - 예측변수로, 선형 회귀 모델에서는 출력층이 단 하나의 뉴런으로 존재한다.

PyTorch에서 선형 회귀 모델을 구축할 때는 `nn.Module` 클래스에서 상속받아 생성할 수 있다.

`nn.Module`은 신경망의 모든 계층을 정의하기 위해 사용되는 기본 클래스로 이를 상속받아 신경망의 각 계층을 정의하고, 여러 계층들을 조합하여 복잡한 신경망 모델을 구축한다.

`nn.Module`의 장점은 일관성, 모듈화과 GPU 지원 뿐만 아니라 자동 미분과 최적화, 디버깅과 로깅 등 모델을 구축하는데 편리한 기능을 제공한다.

상속받는 코드 표현은 다음과 같다.

```python
import torch
import torch.nn as nn

class LinearRegressionModel(nn.Module):  # 클래스
    def __init__(self):     # 생성자 메서드
        super(LinearRegressionModel, self).__init__()   # 속성: 선형 계층 정의
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):   # 메서드: 순전파 연산 정의
        y = self.linear(x)
        return y

model = LinearRegressionModel()     # 인스턴스 생성
```


선형 회귀 모델의 성능은 오차를 이용하여 판단한다.

`오차`는 목표 변수 $t$와 예측 변수 $y$의 차이 $(t - y)$를 의미한다.

<img src="https://curriculum.cosadama.com/machine-learning/3-2/residual2.png">

- 오차가 크다면 가중치 $w$와 바이어스 $b$의 값의 수정이 필요하다.
- 오차가 작다면 가중치 $w$와 바이어스 $b$의 값이 최적화되었다고 판단한다.

따라서, 선형 회귀 모델에서는 오차의 총합이 최소가 되는 $y = wx + b$ 함수의 가중치 $w$와 바이어스 $b$를 찾아야한다.

이러한 오차를 함수로 표현한 것을 `손실 함수(loss function)`라고 부른다.

이 때, 오차의 값들이 양수, 음수가 혼재할 수 있으므로 정확한 측정을 하기 위해서는 **모든 오차에 절대값을 취하거나 오차에 제곱**을 취해 음수를 없애주어야 한다.

특히, 오차의 제곱하는 방식은 오차를 크게 키우기 때문에 학습이 수월해진다. 또한 손실함수가 볼록함수 형태이기 때문에 다양한 최적화 기법을 적용할 수 있다. 

이러한 방법을 `평균제곱오차(Mean Squared Error)`라고 하며 줄여서 `MSE`라고 한다.

$$
\text{Loss Function} = \frac{1}{n} \sum_{i=1}^n [t_i - (wx_i + b)]^2
$$

위의 수식으로 부터 나온 결과값을 줄이는 방향으로 가중치($w$)와 바이어스($b$)를 줄여가야 한다.

이를 PyTorch 코드로 표현하면 `loss_function = nn.MSELoss()`이다.


