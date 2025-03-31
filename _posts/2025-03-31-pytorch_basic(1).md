---
title: "파이토치 기초(1)"
excerpt: "텐서 / 가설 / 손실 함수 / 최적화"

categories:
  - Pytorch
tags:
  - [Pytorch]

permalink: /pytorch/basic-1/

toc: true
toc_sticky: true

date: 2025-03-31
last_modified_at: 2025-03-31
---

## 🦥 텐서

**텐서(Tensor)**란 넘파이 라이브러리의 ndarray 클래스와 유사한 구조로 **배열(Array)**이나 **행렬(Matrix)**과 유사한 자료 구조(자료형)다. 파이토치에서는 텐서를 사용하여 모델의 입출력뿐만 아니라 모델의 매개변수를 부호화(Encode)하고 GPU를 활용해 연산을 가속화할 수 있다.

넘파이와 파이토치<br>
- 공통점: 수학 계산, 선형 대수 연산을 비롯해 전치(Tranposing), 인덱싱(Indexing), 슬라이싱(Slicing), 임의 샘플링(Random Sampling) 등 다양한 텐서 연산을 진행할 수 있다.
- 차이점: 파이토치는 **GPU 가속(GPU Acceleration)**을 적용할 수 있어 CPU 텐서와 GPU 텐서로 나눠지고, 각각의 텐서를 상호 변환하거나 GPU 사용 여부를 설정한다.

텐서의 형태를 시각화하면 다음과 같다.

<img src="https://raw.githubusercontent.com/HiddenBeginner/hiddenbeginner.github.io/master/static/img/_posts/2020-1-21-pytorch_tensor/figure1.png">

- 스칼라(Scalar): 크기만 있는 물리량이지만, 0차원 텐서라고 부른다. 모든 값의 기본 형태로 볼 수 있으며 차원은 없다.
- 벡터(Vector): 스칼라 값들을 하나로 묶은 형태로 간주할 수 있으며 **(N, )**의 차원을 갖는다.
    - [1, 2, 3]과 같은 형태로, 파이썬에서 많이 사용하는 1차원 리스트와 비슷하다. 
- 행렬(Matrix):  벡터값들을 하나로 묶은 형태로 간주할 수 있으며 **(N, M)**으로 표현한다.
    - [[1, 2, 3], [4, 5, 6]]과 같은 형태로 회색조(Grayscale) 이미지를 표현하거나 좌표계(Coordinate System)로도 활용될 수 있다.
- 배열(Array): 3차원 이상의 배열을 모두 지칭하며, 각각의 차원을 구별하기 위해 N 차원 배열 또는 N 차원 텐서로 표현한다. 즉, 행렬을 세 개 생성해 겹쳐 놓은 구조로 볼 수 있다.
    - 배열의 경우 이미지를 표현하기에 가장 적합한 형태를 띤다.
    - 이미지의 경우 (C, H, W)로 표현하며, C는 채널, H는 이미지의 높이, W는 이미지의 너비가 된다.
- 4차원 배열: 3차원 배열들을 하나로 묶은 형태이므로 이미지 여러 개의 묶음으로 볼 수 있다. 파이토치를 통해 이미지 데이터를 학습시킬 때 주로 4차원 배열 구조의 형태로 가장 많이 사용한다. 
    - 이미지의 경우 (N, C, H, W)로 표현한다. N의 경우 이미지의 개수를 의미한다.

### 텐서 생성

텐서 생성 방법은 `torch.tensor()` 또는 `torch.Tensor()`로 생성할 수 있다.
- `torch.tensor()` : **입력된 데이터를 복사해 텐서로 변환**하는 함수이다. 즉, 데이터를 복사하기 때문에 값이 무조건 존재해야 하며 입력된 데이터 형식에 가장 작합한 텐서 자료형으로 변환한다.
- `torch.Tensor()` : **텐서의 기본형으로 텐서 인스턴스를 생성하는 클래스**다. 인스턴스를 생성하기 때문에 값을 입력하지 않는 경우 비어 있는 텐서를 생성한다.

가능한 자료형이 명확하게 표현되는 클래스 형태의 `torch.Tensor()`를 사용하는 것을 권장한다.

```python
# 텐서 생성
import torch

print(torch.tensor([1, 2, 3]))
print(torch.Tensor([[1,2, 3], [4, 5, 6]]))
print(torch.LongTensor([1, 2, 3]))
print(torch.FloatTensor([1, 2, 3]))

# tensor([1, 2, 3])
# tensor([[1., 2., 3.],
#         [4., 5., 6.]])
# tensor([1, 2, 3])
# tensor([1., 2., 3.])
```

- `torch.tensor()`는 자동으로 자료형을 할당하므로 입력된 데이터 형식을 참조해 Int 형식으로 할당
- `torch.Tensor()`는 입력된 데이터 형식이 Int형이지만 Float 형식으로 생성됐는데, 이는 기본 유형이 Float이므로 소수점 형태로 변환
- `torch.LongTensor()`, `torch.FloatTensor()`, `torch.IntTensor()` 모두 데이터 형식이 미리 선언된 클래스다.

### 텐서 속성

텐서의 속성은 크게 **형태(shape), 자료형(dtype), 장치(device)**가 존재한다.
- 형태: 텐서의 차원을 의미
- 자료형: 텐서에 할당된 데이터 형식
- 장치: 텐서의 GPU 가속 여부를 의미

텐서 연산을 진행할 때 위의 세 가지 속성이 모두 맞아야 작동이 된다.

```python
import torch

tensor = torch.rand(1, 2)

print(tensor)
print(tensor.shape)
print(tensor.dtype)
print(tensor.device)

# tensor([[0.8522, 0.3964]])
# torch.Size([1, 2])
# torch.float32
# cpu
```

### 차원 변환

차원 변환은 가장 많이 사용되는 메서드 중 하나로 머신러닝 연산 과정이나 입출력 변환 등에 많이 활용된다.

```python
# 텐서 차원 변환
import torch

tensor = torch.rand(1, 2)
print(tensor)
print(tensor.shape)

tensor = tensor.reshape(2, 1)
print(tensor)
print(tensor.shape)

# tensor([[0.6499, 0.3419]])
# torch.Size([1, 2])
# tensor([[0.6499],
#         [0.3419]])
# torch.Size([2, 1])
```

텐서의 차원 변환은 `reshape()` 메서드를 활용할 수 있다.

### 자료형 설정

텐서에 있어서 자료형은 가장 중요한 요소다. 

```python
# 텐서 자료형 설정
import torch

tensor = torch.rand((3, 3), dtype=torch.float)
print(tensor)

# tensor([[0.6837, 0.7457, 0.9212],
#         [0.3221, 0.9590, 0.1553],
#         [0.7908, 0.4360, 0.7417]])
```

텐서의 자료형 설정에 입력되는 인수는 `torch.*` 형태로 할당한다.
- `torch.float`는 32비트 부동 소수점 형식을 갖지만, `float`은 64비트 부동 소수점을 갖는다. 이는 메모리 필요를 줄일 수 있다.

### 장치 설정

장치 설정을 정확하게 할당하지 않으면 **실행 오류(Runtime Error)**가 발새하거나 CPU 연산이 되어 학습하는 데 오랜 시간이 소요된다. 그러므로 모델 학습을 하기 전에 장치 설정을 확인하는 과정이 필요하다.

```python
# 텐서 GPU 장치 설정
import torch 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cpu = torch.FloatTensor([1, 2, 3])
gpu = torch.cuda.FloatTensor([1, 2, 3])
tensor = torch.rand((1, 1), device=device)
print(device)
print(cpu)
print(gpu)
print(tensor)

# cuda
# tensor([1., 2., 3.])
# tensor([1., 2., 3.], device='cuda:0')
# tensor([[0.1998]], device='cuda:0')
```

- `torch.cuda.is_available()`: CUDA 사용 여부를 확인할 수 있는 함수

애플 실리콘에서는 MPS를 통한 GPU 가속을 적용한다.
- `device = "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu"`

### 장치 변환

CPU 장치를 사용하는 텐서와 GPU 장치를 사용하는 텐서는 상호 간 연산이 불가능하다. 하지만 CPU 장치를 사용하는 텐서와 넘파이 배열 간 연산은 가능하며, GPU 장치를 사용하는 텐서와 넘파이 배열 간 연산은 불가능하다.

넘파이 배열 데이터를 학습에 활용하려면 GPU 장치로 변환해야한다.

```python
# 텐서 장치 변환
import torch

cpu = torch.FloatTensor([1, 2, 3])
gpu = cpu.cuda()
gpu2cpu = gpu.cpu()
cpu2gpu = cpu.to("cuda")
print(cpu)
print(gpu)
print(gpu2cpu)
print(gpu2cpu)
print(cpu2gpu)

# tensor([1., 2., 3.])
# tensor([1., 2., 3.], device='cuda:0')
# tensor([1., 2., 3.])
# tensor([1., 2., 3.], device='cuda:0')
```

**장치(device)** 간 상호 변환은 cuda와 cpu 메서드를 통해 할 수 있다.
- `cuda()`: CPU 장치로 선언된 값을 GPU로 변환 가능
- `cpu()`: GPU 장치로 선언된 값을 CPU로 변환
- `to()`: 장치를 간단하게 변환(MPS 장치로 변환도 가능)

### 넘파이 배열의 텐서 변환

넘파이나 다른 라이브러리의 데이터를 파이토치에 활용하려면 텐서 형식으로 변환해야 한다.

```python
# 넘파이 배열의 텐서 변환
import torch
import numpy as np

ndarray = np.array([1, 2, 3], dtype=np.uint8)
print(torch.tensor(ndarray))
print(torch.Tensor(ndarray))
print(torch.from_numpy(ndarray))

# tensor([1, 2, 3], dtype=torch.uint8)
# tensor([1., 2., 3.])
# tensor([1., 2., 3.])
# tensor([1, 2, 3], dtype=torch.uint8)
```

### 텐서의 넘파이 배열 변환

텐서를 넘파이 배열로 변환하는 방법은 추론된 결과를 후처리하거나 결과값을 활용할 때 주로 사용된다.

```python
# 텐서의 넘파이 배열 변환
import torch

tensor = torch.cuda.FloatTensor([1, 2, 3])
ndarray = tensor.detach().cpu().numpy()
print(ndarray)
print(type(ndarray))

# [1, 2, 3,]
# <class 'numpy.ndarray'>
```

텐서는 기존 데이터 형식과 다르게 학습을 위한 데이터 형식으로 모든 연산을 추적해 기록한다. 이 기록을 통해 **역전파(Backpropagation)** 등과 같은 연산이 진행돼 모델 학습이 이뤄진다.

텐서를 넘파이 배열로 변환할 때는 `detach()` 메서드를 적용한다. 이 메서드는 현재 연산 그래프에서 분리된 새로운 텐서를 반환한다. 

## 🦥 가설

## 🦥 손실 함수

## 🦥 최적화

