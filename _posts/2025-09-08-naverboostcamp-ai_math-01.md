---
title: "벡터(Vector)"
description: "벡터의 무엇인지 정의하고 기초적인 연산 노름(Norm), 벡터의 거리와 내적에 대한 내용을 정리한 포스트입니다."

categories: [Math for AI, Linear Algebra]
tags: [linear-algebra, Vector]

permalink: /naver-boostcamp/linear-algebra/01

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-09-08
last_modified_at: 2025-09-08
---

벡터는 숫자가 원소인 `자료형(data type)`으로 `배열(array)`이라 부른다.

- 행벡터

    $$
    \mathbf{x} = [x1, x2, \ldots, x_d]
    $$

- 열벡터

    $$
    \mathbf{x} = \begin{bmatrix}
    x1 \\
    x2 \\
    \vdots \\
    x_d
    \end{bmatrix}
    $$

보통 수식은 열벡터를 기본형으로 쓰지만 필요에 따라 행벡터를 혼용해서 쓴다.

- 차원(dimension): 벡터에 저장된 원소의 개수를 말한다.
    - dim($\mathbf{x}$) = d

파이썬에서는 `numpy.array`로 구현한다.

```python
import numpy as np

# 수식: x = [1, 7, 2]
x = np.array([1, 7, 2])
```

벡터는 공간에서 `한 점`으로 표현되며 원점으로부터 상대적 `위치`를 표현한다.

<img src="https://velog.velcdn.com/images/guts4/post/9814de56-1ef6-4968-a19d-17c6c4106d7a/image.png">


## 벡터의 연산
---------

이때, 벡터는 숫자를 곱헤주면 **방향은 변하지 않고 길이만 변하는데**, 이를 `스칼라곱`이라고 부른다.

$$
\alpha \mathbf{x} = [\alpha x_1, \ldots, \alpha x_d]
$$

벡터끼리 **차원이 같으면** 덧셈, 뺄셈, 성분곱(Elementwise product) 계산이 가능하다.

$$
\begin{align*}
\mathbf{x} \pm \mathbf{y} = [x_1 \pm y_1, \ldots, x_d \pm y_d] \\
\mathbf{x} \odot \mathbf{y} = [x_1 \odot y_1, \ldots, x_d \odot y_d]
\end{align*}
$$

수식을 numpy로 구현하면 다음과 같다.

```python
import numpy as np

x = np.array([1,7, 2])
y = np.array([5, 2, 1])

# 덧셈
x + y

# 뺄셈
x - y

# 성분곱
x * y
```

- 두 벡터의 덧셈은 다른 벡터로부터 `상대적 위치이동`을 표현한다.
- 두 뺄셈은 덧셈의 반대 방향으로 움직인다.

## 벡터의 노름
-----------

벡터의 노름(norm)은 `원점으로부터 거리`를 말하며 `벡터의 크기`를 나타낸다.

- $L_1$-norm : 각 성분의 `변화량의 절대값`을 모두 더한다.

    $$
    \lVert \mathbf{x} \rVert _1 = \sum_{i=1}^d |x_i|
    $$

- $L_2$-norm : 피타고라스 정리를 이용해 `유클리드 거리`를 계산한다. 

    $$
    \lVert \mathbf{x} \rVert _2 = \sqrt{\sum_{i=1}^d |x_i|^2}
    $$


`numpy.linalg.norm`을 함수를 이용해 노름을 구할 수 있다.

서로 다른 노름을 사용하는 이유는 노름의 종류에 따라 `기하학적 성질`이 달라지기 때문이다.

<img src="https://velog.velcdn.com/images/guts4/post/4d1e2a50-21a3-424b-85bd-311a85fd9b93/image.png" width="500" height="300">

$L_1$-norm은 Robust 학습이나 Lasso 회귀에서 주로 사용하며 $L_2$-norm은 Laplace 근사, Ridge 회귀에서 사용된다.

## 두 벡터 사이의 거리
--------------

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTtktVQlgfnMN7pgnh-hMpMDwPJ8EGje9_pjTFcTTn3YnEWeQRXMMjq-4S0Y_0FDwXBjk0&usqp=CAU">

노름을 이용하면 두 벡터 사이의 거리를 계산할 수 있다. `벡터 사이의 거리`는 두 점 $\mathbf{x}$, $\mathbf{y}$ 사이의 거리를 의미하며 `벡터의 뺄셈`을 이용한다.



## 두 벡터 사이의 각도
---------------

<img src="https://velog.velcdn.com/images/guts4/post/afb22f10-cc42-41b0-b00a-c1bd879e3ee4/image.png" width="500" height="300">

$L_2$-norm과 `제 2 코사인 법칙`을 이용하면 두 벡터 사이의 각도를 계산할 수 있다.

이 계산에서 분자를 쉽게 계산하는 방법으로 `내적`을 이용하면 된다.

<img src="https://velog.velcdn.com/images%2Fwhattsup_kim%2Fpost%2Fce84f1fd-a11f-4413-962f-8720c294359c%2Fimage.png" width="400" height="250">

- 내적(inner product) : 두 벡터의 각 성분들끼리 곱한 후 모두 더한 값을 의미한다.
    - `np.inner`을 이용해 계산한다.

```python
def theta(x, y):
    v = np.inner(x, y) / (l2_norm(x) * l2_norm(y))
    theta = np.arccos(v)
    return theta
```

## 내적
------------

내적은 `정사영(orthogonal projection)된 벡터의 길이`와 관련이 있다.

<img src="https://velog.velcdn.com/images%2Fwhattsup_kim%2Fpost%2F6cdf82ab-9564-4381-8e94-4bc48e6a08ae%2Fimage.png">

이때 `Proj(x)`의 길이는 코사인법칙에 의해 $\lVert \mathbf{x} \rVert cos \theta$이다.

즉, 내적은 정사영의 길이를 **벡터 $\mathbf{y}$의 길이 \lVert \mathbf{y} \rVert$만큼 조정**한 값이다.

<img src="https://velog.velcdn.com/images%2Fwhattsup_kim%2Fpost%2Fffd4c74b-a071-4a41-b181-aa6293727824%2Fimage.png">

내적은 두 벡터의 `유사도(similarity)`를 측정하는데 사용할 수 있다.