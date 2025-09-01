---
title: "Tensor 생성과 조작"
description: "네이버 부스트캠프 PyTorch 강의 정리 포스트입니다."

categories: [Naver-Boostcamp, PyTorch]
tags: [Naver-Boostcamp, pytorch, Tensor]

permalink: /naver-boostcamp/PyTorch/pytorch-02/

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-09-01
last_modified_at: 2025-09-01
---

## Tensor의 생성
-------------

Tensor를 생성하는 방법은 굉장히 다양하다.

### 특정한 값으로 초기화된 Tensor 생성 & 변환

Tensor을 생성할 때 특정한 값을 정하여 초기화할 수 있다.

- 0으로 초기화된 Tensor를 생성하는 표현
    - `torch.zeros([2, 3, 4])`
- 1로 초기화된 Tensor를 생성하는 표현
    - `torch.ones([1, 2, 5])`

생성 뿐만 아니라 특정한 값으로 변환하여 Tensor를 생성하는 것도 가능하다.

- 크기와 자료형이 같은 0으로 초기화된 Tensor로 변환하는 표현
    - `torch.zeros_like(e)`
- 크기와 자료형이 같은 1로 초기화된 Tensor로 변환하는 표현
    - `torch.ones_like(b)`


```python
import torch

# 0으로 초기화된 Tensor 생성
torch.zeros(5)  # tensor([0., 0., 0., 0., 0.])

# 1로 초기화된 Tensor 생성
torch.ones(3)   # tensor([1., 1., 1.])

a = torch.ones([3, 2])
b = torch.zeros([2, 3])

# 크기와 자료형이 같은 0으로 초기화된 Tensor로 변환
torch.zeros_like(a)
"""
tensor([[0., 0.],
       [0., 0.],
       [0., 0.]])
"""

# 크기와 자료형이 같은 1로 초기화된 Tensor로 변환
torch.ones_like(b)
"""
tensor([[1., 1., 1.],
       [1., 1., 1.]])
"""
```

### 난수로 초기화된 Tensor 생성 & 변환

특정한 값이 아니라 난수로 초기화된 Tensor를 생성할 수도 있다.

- [0, 1] 구간의 연속균등분포 난수 Tensor 생성
    - `torch.rand([2, 3])`
    - 파라미터 초기화를 하는 경우
    - 무작위 데이터를 생성하는 경우

> 연속균등분포: 특정한 두 경계값 사이의 모든 값에 대해 동일한 확률을 가지는 확률분포르 두 경계값은 [0, 1] 사이에 존재해야한다.<br>
>
> $$f(x) = \frac{1}{b-a} (a \le x \le b)$$
>
> 연속균등분포의 모수는 다음과 같다.
> - 평균(기대값): $\mu = \frac{a + b}{2}$
> - 분산: $\sigma^2 = \frac{(a-b)^2}{12}$
> - 표준편차: $\sigma = \sqrt{\frac{(a-b)^2}{12}} = \frac{a-b}{2\sqrt{3}}$
>
> <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQBZbjhOArCyA2aDFkfQfTmLTCev3MazlXeNA&s">

- 표준정규분포에서 추출한 난수로 Tensor 생성
    - `torch.randn([3, 2])`

> 표준정규분포: 평균이 0이고 표준편차가 1인 종모양의 곡선<br>
> 
> $$f(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}}$$
> - 평균(기대값): $\mu = 0$
> - 분산: $\sigma^2 = 1$
> - 표준편차: $\sigma = 1$
>
> <img src="https://magmatart.dev/assets/posts/images/UDL15/figure1.PNG" width="300" height="300">

생성뿐만 아니라 변환하는 함수또한 존재한다.

- 크기와 자료형이 같은 연속균등분포 난수 Tensor로 변환
    - `torch.rand_like(k)`
- 크기와 자료형이 같은 표준정규분포 난수 Tensor로 변환
    - `torch.randn_like(i)`

```python
# 연속균등분포 난수 Tensor
torch.rand(3)   # tensor([0.9701, 0.3271, 0.2040])

# 표준정규분포 난수 Tensor
torch.randn(3)  # tensor([-0.8471, 0.8292, -0.4368])

k = torch.randn(3)
i = torch.rand(3)

# 크기와 자료형이 같은 연속균등분포 난수 Tensor로 변환
torch.rand_like(k)  # tensor([0.8607, 0.4303, 0.1227])

# 크기와 자료형이 같은 표준정규분포 난수 Tensor로 변환
torch.randn_like(i) # tensor([-0.1718, -0.5145, 0.7640])
```

### 지정된 범위 내에서 초기화된 Tensor 생성

지정된 범위를 정하여 일정한 간격을 가지는 Tensor를 생성할 수 있다.

- `torch.arange(start=1, end=4, step=0.5)`

### 초기화되지 않은 Tensor 생성

초기화 되지 않은 Tensor를 생성하는 방법도 존재한다.

여기서 초기화되지 않은 Tensor는 생성된 Tensor의 각 요소가 명시적으로 다른 특정한 값으로 설정되지 않았음을 의미하고 해당 Tensor는 메모리에 이미 존재하는 임의의 값들로 채워진다.

초기화되지 않은 Tensor을 사용한는 이유는 다음과 같다.

- 성능 향상: 초기 값을 설정하는 단계는 불필요한 자원을 소모
- 메모리 최적화: 불필요한 초기화는 메모리 사용량을 증가시키므로 필요한 계산만 사용하여 메모리 효율성을 높일 수 있음

대표적인 방법은 다음과 같다.

- `torch.empty(5)`

위의 방법처럼 생성된 초기화 되지 않은 Tenso는 `torch.fill_(3.0)` 함수를 호출해 요소를 채울 수 있다.

### list, Numpy 데이터로부터 Tensor 생성

Python의 List와 Numpy 데이터로부터 Tensor을 생성할 수 있다.

- List 데이터로부터 Tensor 생성
    - `torch.tensor(l)`
- Numpy 데이터로부터 Tensor 생성
    - `torch.from_Numpy(u).float()`
    - `float()`를 붙이는 이유는 기본적으로 생성된 데이터가 정수형이기 때문에 변환이 필요함

> List와 Numpy 차이점: List는 가변적인 컨테이너 데이터타입이며 연산 및 조작에 적합하지 않은 반면 Numpy는 대규모 다차원 배열 연산이 가능하다.

```python
import numpy as np

s = [1, 2, 3, 4, 5]
u = np.array([[0, 1],
              [2, 3]])

# list를 Tensor로 변환
torch.tensor(s)    # tensor([1, 2, 3, 4, 5])

# Numpy를 Tensor로 변환
torch.from_numpy(u).float()
"""
tensor([[0., 1.,],
        [2., 3.]])
"""
```

### CPU Tensor 생성

- 정수형 CPU Tensor 생성
    - `torch.IntTensor([1, 2, 3, 4])`
- 실수형 CPU Tensor 생성
    - `torch.FloatTensor([1, 2, 3, 4])`
- 8비트 부호 없는 정수형 CPU Tensor 생성: `torch.ByteTensor`
- 16비트 부호 있는 정수형 CPU Tensor 생성: `torch.CharTensor`
- 64비트 부호 있는 정수형 CPU Tensor 생성: `torch.LongTensor`
- 64비트 부호 있는 실수형 CPU Tensor 생성: `torch.DoubleTensor`

### Tensor의 복제

Tensor를 복제하는 코드 표현으로 `clone()`과 `detach()`가 있다. 차이점은 `detach()`는 **계산 그래프에서 분리**하여 새로운 Tensor을 얻는다는 것이다.

```python
import torch

x = torch.tensor([1, 2, 3, 4, 5])

y = x.clone()
z = x.detach()
```

### CUDA Tensor 생성과 변환


GPU: 그래픽 처리 장치를 의미하며 대규모 데이터 처리와 복잡한 계산을 위해 사용된다.

- 데이터가 CPU에 있는지 GPU에 있는지를 확인할 수 있다.
    - `a.device`
- 현재 환경이 CUDA 기술을 사용할 수 있는지 확인하기 위해서는 `torch.cuda.is_available()`로 확인이 가능하다.
- CUDA 이름을 확인할 때는 `torch.cuda.get_device_name(device=0)`를 사용한다.
- Tensor를 GPU에 할당할 때 쓰는 코드는 다음과 같다.
    - `torch.tensor([1, 2, 3, 4]).to('cuda')`
    - `torch.tensor([1, 2, 3, 4]).cuda()`
- GPU에 할당된 Tensor를 CPU Tensor로 변환하는 코드는 다음과 같다.
    - `b.to(device='cpu')`
    - `b.cpu()`

## Tensor의 indexing & slicing
--------------











