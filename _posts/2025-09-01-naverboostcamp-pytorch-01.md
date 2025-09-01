---
title: "PyTorch 기초"
description: "네이버 부스트코스의 Pre-course 강의를 기반으로 작성한 포스트입니다."

categories: [Naver-Boostcamp, PyTorch]
tags: [Naver-Boostcamp, pytorch, Tensor]

permalink: /naver-boostcamp/PyTorch/pytorch-01/

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-09-01
last_modified_at: 2025-09-01
---

## PyTorch Introduction
-------------

**PyTorch란?**

PyTorch는 간편한 딥러닝 API를 제공하여 딥러닝 알고리즘을 구현하고 실행하기 위해 만들어진 딥러닝 프레임워크이다.

> API: Application Programming Interface의 줄임말로 응용 프로그램이 서로 상호작용하는데 사용하는 명령어, 함수, 프로토콜의 집합을 의미한다.

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR3dWwDbb-8VqQxRHt8K3gNT5uItv6Fkyxur3bN2A9kqWyzsgjfZ2vq3_X68okEh67jwIQ&usqp=CAU">

파이토치의 장점은 다음과 같다.

- 확장성이 뛰어나 다양한 규모의 프로젝트에 적응할 수 있도록 작업량을 처리할 수 있는 능력이 좋으며 멀티플랫폼 프로그래밍 인터페이스이므로 다양한 환경에서 사용할 수 있다.
- Pythonic한 특징이 있어 배우기 쉬우며 API를 제공하기 때문에 간편한다.
- 활발한 커뮤니티와 Hugging Face와 같은 풍부한 모델 구현체를 가지고 있다.
- 동적 계산 그래프로 정적 계산 그래프에 비해 코드의 구현 길이가 짧다.
- GPU 지원

## Tensor
------------

`Tensor`는 PyTorch의 핵심 데이터 구조로 Numpy의 다차원 배열과 유사한 형태로 데이터를 표현한다.

### Tensor의 표현

- 0-D Tensor: 스칼라(Scalar), 하나의 숫자로 표현되는 양이다.
- 1-D Tensor: 벡터(Vector), 순서가 지정된 여러 개의 숫자들이 일렬로 나영된 구조이다.
- 2-D Tensor: 행렬(Matrix), 동일한 크기의 1D Tensor들이 모여서 형성한, 행과 열로 구성된 사각형 구조
    - ex. 그레이 스케일 이미지
- 3-D Tensor: 동일한 크기의 2D Tensor들이 여러 개 쌓여 형성된 입체적인 배열 구조
    - ex. 컬러 이미지
- N-D Tensor(N $\ge$ 4): 동일한 크기의 (N-1) D Tensor들이 여러 개 쌓여 형성된 입체적인 배열 구조
    - 4-D Tensor의 영상

<img src="https://miro.medium.com/1*MQIAAntN5tYgKEDNTcpZmg.png">

```python
import torch

# 0-D Tensor
a = torch.tensor(36.5)

# 1-D Tensor
b = torch.tensor([1, 4, 2, 0.9, -0.2])

# 2-D Tensor
c = torch.tensor([[3, 54, 23, 1],
                 [13, 43, 3, 14],
                 [45, 23, 194, 92]])

# 3-D Tensor
d = torch.tensor([[[255, 0, 0],
                   [0, 255, 0]],
                  [[0, 0, 255],
                   [0, 255, 0]]])
```

## Data type
----------

PyTorch에서의 데이터 타입(dtype)은 Tensor가 저장하는 값의 데이터 유형을 의미함

### 정수형 데이터 타입

정수형 데이터 타읍은 **소수 부분이 없는 숫자를 저장**하는 데 사용되는 데이터 타입으로 5가지의 유형으로 구분한다.

- 8비트 부호 없는 정수: 8개의 이진 비트를 사용하여 0부터 255까지의 정수를 표현
    - `dtype = torch.uint8`
- 8비트 부호 있는 정수: 8개의 이진 비트를 사용하여 -128부터 127까지의 정수를 표현
    - `dtype = torch.int8`
    - 맨 처음 비트를 부호를 표현하는데 사용한다.(0일 땐 '+' / 1일 땐 '-')
- 16비트 부호 있는 정수: 16개의 이진 비트를 사용하여 -32,768부터 32,767까지 정수를 표현
    - `dtype = torch.int16` or `dtype = torch.short`
- 32비트 부호 있는 정수: 32개의 이진 비트를 사용하여 -2,147,483,648부터 2,147,483,647까지의 정수를 표현
    - 대부분 프로그래밍에서  **표준적인 정수 크기로 사용**
    - `dtype = torch.int32` or `dtype = torch.int`
- 64비트 부호 있는 정수: 64개의 이진 비트를 사용하여 -9,223,372,036,854,775,808부터 9,223,372,036,854,775,807까지의 정수를 표현
    - `dtype = torch.int64` or `dtype = torch.long`

```python
a = torch.tensor(234, dtype=torch.int16)
```

> *if `uint8`에 부호 '-'가 있는 경우는 어떻게 출력될까?*<br>
> `print(torch.tensor(-1, dtype=torch.uint8))` 를 출력하면 255로 출력된다.

### 실수형 데이터 타입

실수형 데이터 타입은 32비트 부동 소수점 수와 64비트 부동 소수점 수의 유형으로 구분한다.

**실수형 데이터 타입들은 신경망의 수치 계산**에서 사용되는 중요한 데이터 타입이다.

- 32비트 부동 소수점 수: 32개의 이진 비트를 사용하여 가수부와 지수부로 표현
    - `dtype = torch.float32` or `dtype = torch.float`
- 64비트 부동 소수점 수: 64개의 이진 비트를 사용하여 가수부와 지수부로 표현
    - `dtype = torch.float64` or `dtype = torch.double`

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTkteP806Yh620KD7tDe1xRhDPAFPpjEm2IcsAs86azJVMcKwgAAAsNb_kuA-dgkrqDBuY&usqp=CAU">

```python
g = torch.tensor(1, dtype=torch.float32)
```

> **고정 소수점이 아닌 부동 소수점 방식으로 선택한 이유**
>
> 고정 소수점은 소수부를 각 자리마다 따로 저장해야하므로 4bit(0~9까지 저장)가 필요하므로 컴퓨터 메모리를 많이 차지하는 단점이 있다. 
> 반면에, 부동 소수점은 숫자를 정규화하여 가수부와 지수부로 나누어 표현하면 메모리를 적게 차지한다.
> <img src="https://devocean.sk.com/editorImg/2023/9/13/2a00158875a5603949cd7ddba1eb279cad1c5e3684bfdb5ed24d6b17dd90d340">

### 타입 캐스팅

PyTorch에서 타입 캐스팅은 **한 데이터 타입을 다른 데이터 타입으로 변환**하는 것을 의미한다.

- 32비트 부동 소수점 수로 변환
- 64비트 부동 소수점 수로 변환

```python
a = torch.tensor([2, 3, 5], dtype = torch.int8)

# 32비트 실수형으로 변환
b = a.float()

# 64비트 실수형으로 변환
c = a.double()
```

## Basic Functions
---------------

PyTorch에는 Tensor의 요소를 반환하거나 계산하는 함수, 특성을 확인하는 메서드가 구현되어있다.

요소를 반환하거나 계산하는 함수는 다음과 같다.

- `min()` : Tensor의 모든 요소들 중 최소값을 반환하는 함수
- `max()` : Tensor의 모든 요소들 중 최대값을 반환하는 함수
- `sum()` : Tensor의 모든 요소들의 합을 계산하는 함수
- `prod()` : Tensor의 모든 요소들의 곱을 계산하는 함수
- `mean()`: Tensor의 모든 요소들의 평균을 계산하는 함수
- `var()`: Tensor의 모든 요소들의 표본분산을 계산하는 함수
- `std()`: Tensor의 모든 요소들의 표본표준편차를 계산하는 함수


```python
import torch

M = torch.tensor([[1, 2, 3],
                  [3, 4, 5]], dtype = torch.float)
# min 함수
torch.min(M)    # tensor(1.)

# max 함수
torch.max(M)    # tensor(5.)

# sum 함수
torch.sum(M)    # tensor(18.)

# prod 함수
torch.prod(M)   # tensor(360.)

# mean 함수
torch.mean(M)   # tensor(3.)

# var 함수
torch.var(M)    # tensor(2.)

# std 함수
torch.std(M)    # tensor(1.4142)
```

> **표본분산이란?**
>
> 표본분산은 주어진 표본 데이터 집합의 분포 정도를 나타내는 통계량으로서, 데이터가 평균값을 중심으로 얼마나 퍼져 있는지를 묘사한다.<br> 
> - 표본표준 편차는 표본분산값에 루트를 씌워 제곱근을 만들면 된다.<br>
>
> 표본분산은 데이터 값들이 평균에서 얼마나 떨어져 있는지를 제곱한 값의 평균으로 계산하며 수식은 아래와 같다.<br>
>
> $$S^2 = \frac{1}{n - 1} \sum_{i = 1}^{n}(x_i - \bar{x})^2$$


Tensor의 특성을 확인하는 메서드는 아래와 같다.

- `I.dim()` : Tensor 'I'의 차원의 수를 확인
- `I.size()` : Tensor 'I'의 크기(모양)을 확인 (`I.shape`라는 속성을 이용해서 확인 가능)
- `I.numel()` : Tensor 'I'에 있는 요소의 총 개수를 확인

```python
import torch

M = torch.tensor([[1, 2, 3],
                  [3, 4, 5]], dtype = torch.float)

# dim 함수
M.dim()     # 2

# size 함수
M.size()    # torch.Size([2, 3])

# numel 함수
M.numel()   # 6
```

