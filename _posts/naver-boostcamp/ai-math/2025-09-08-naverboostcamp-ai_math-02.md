---
title: "[BoostCamp AI Tech / AI Math] 행렬(Matrix)"
description: "행렬의 기본 개념과 선형 변환, 역행렬 그리고 이를 이용한 연립방정식 해를 구하는 방법을 정리한 포스트입니다."

categories: [NAVER BoostCamp AI Tech, AI Math]
tags: [linear-algebra, matrix]

permalink: /naver-boostcamp/linear-algebra/02

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-09-08
last_modified_at: 2026-11-11
---

행렬(matrix)은 벡터를 원소를 가지는 `2차원 배열`이다.

$$
\mathbf{X} = \begin{bmatrix}
1 & -2 & 3 \\
7 & 5 & 0 \\
-2 & -1 & 2
\end{bmatrix}
$$

이를 코드로 표현하면 다음과 같다.

```python
import numpy as np

X = np.array([[1, -2, 3],
              [7, 5, 0],
              [-2, -1, 2]])
```

행렬을 일반화해서 표현하면 아래와 같다.

$$
\mathbf{X} = \begin{bmatrix}
\mathbf{x}_1 \\
\mathbf{x}_2 \\
\vdots \\
\mathbf{x}_1 \\
\end{bmatrix} = \begin{bmatrix}
x_{11} & x_{12} & \cdots & x_{1m} \\
x_{21} & x_{22} & \cdots & x_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
x_{n1} & x_{n2} & \cdots & x_{nm} \\
\end{bmatrix}
$$

이때, 행렬의 원소는 각각 `행(row)`과 `열(column)`이라는 인덱스(index)로 표현한다.

행렬을 $\mathbf{X} = (x_{ij})$ 라고도 표기한다.

> 전치 행렬(transpose matrix) : 행과 열의 인덱스가 바뀐 행렬
> - $\mathbf{X}^T = (x_{ij})^T = (x_{ji})$

벡터가 공간에서 한 점을 의미한다면 행렬은 `여러 점들`을 나타낸다. 이때 행렬의 행벡터 $\mathbf{x}_ i$는 $i$**번째 데이터**를 의미하고 $x_{ij}$ 는 $i$**번째 데이터의** $j$**번쩨 변수의 값**을 말한다.

즉, 행렬은 어떤 데이터를 표현하는데 사용할 수 있다.

## 행렬 표기법
------------

- $A \in \mathbb{R}^{n \times n}$: Square matrix
    - row 개수 = column 개수
    - e.g., 
        $$
        A = \begin{bmatrix}
        1 & 6 \\
        3 & 4
        \end{bmatrix}
        $$
- $A \in \mathbb{R}^{m \times n}$: Rectangular matrix
    - row 개수 $\neq$ column 개수 도 가능
    - e.g., 
        $$
        A = \begin{bmatrix}
        1 & 6 \\
        3 & 4 \\
        5 & 2
        \end{bmatrix}
        $$
- $A^T$: Transpose Matrix
    - e.g.,
        $$
        A = \begin{bmatrix}
        1 & 3 & 5 \\
        6 & 4 & 2 \\
        \end{bmatrix}
        $$

- $A_{ij}$ : 행렬 $A$의 $i$ 번째 행, $j$ 번째 열의 요소
- $A_{i,:}$ : 행렬 $A$의 $i$ 번째 행 벡터
- $A_{:,j}$ : 행렬 $A$의 $j$ 번째 열 벡터

## 행렬의 덧셈, 뺄셈, 성분곱, 스칼라 곱
------------

행렬끼리 **같은 모양을 가지면** 덧셈, 뺄셈을 계산할 수 있다.

$$
\mathbf{X} \pm \mathbf{Y} = (x_{ij} \pm y_{ij})
$$

행렬의 성분곱 또한 벡터와 같이 각 인덱스 위치끼리 곱한다.

$$
\mathbf{X} \odot \mathbf{Y} = (x_{ij} \ y_{ij})
$$

스칼라곱 또한 벡터와 동일하게 모든 성분에 똑같이 숫자를 곱해준다.

$$
\alpha \mathbf{X} = (\alpha x_{ij})
$$

## 행렬 곱셈
-----------

행렬 곱셈(matrix multiplication)은 $i$ **번째 행벡터와** $j$ **번째 열벡터 사이의 내적**을 성분으로 가지는 행렬을 계산한다.

이때, 행렬곱은 $\mathbf{X}$의 열의 개수와 $\mathbf{Y}$의 행의 개수가 같아야 한다.

$$
\mathbf{X}\mathbf{Y} = \bigg( \sum_{k} x_{ik} \ y_{kj} \bigg)
$$

`numpy`에선 `@` 연산을 사용한다.

`@`는 내부적으로 `np.matmul`을 호출한다. 비슷해 보이는 `np.dot`과는 다음이 다르다.

- `np.dot`은 스칼라곱이 허용되지만 `np.matmul`은 허용되지 않는다.
- 2차원 행렬에서는 결과가 같지만, 3차원 이상의 배열(텐서)에서는 결과가 달라진다. `np.dot`은 두 배열의 **내적곱(dot product)**, `np.matmul`은 **행렬곱(matrix product)**이다.

### 행렬 곱셈의 성질

- 행렬은 교환법칙(commutative)이 성립하지 않는다.

    - $A \in \mathbb{R}^{2 \times 3}$ 이며 $B \in \mathbb{R}^{3 \times 5}$ 인 경우, $AB$는 가능하지만 $BA$ 계산은 불가능하다.
    - $A \in \mathbb{R}^{2 \times 3}$ 이며 $B \in \mathbb{R}^{3 \times 2}$ 인 경우, $AB$, $BA$ 계산이 모두 가능하지만 결과로 나오는 행렬의 크기가 다르다.
    - $A \in \mathbb{R}^{2 \times 2}$ 이며 $B \in \mathbb{R}^{2 \times 2}$ 인 경우, $AB$, $BA$ 계산이 모두 가능하고 결과로 나오는 행렬의 크기도 같지만 결과로 나온 행렬의 원소값이 다를 수 있다.
        - 
        $$
        \begin{bmatrix}
        1 & 2 \\
        3 & 4
        \end{bmatrix}
        \begin{bmatrix}
        5 & 6 \\
        7 & 8
        \end{bmatrix} = 
        \begin{bmatrix}
        19 & 22 \\
        43 & 50
        \end{bmatrix}
        $$
        <br>
        $$
        \begin{bmatrix}
        5 & 6 \\
        7 & 8
        \end{bmatrix}
        \begin{bmatrix}
        1 & 2 \\
        3 & 4
        \end{bmatrix}=
        \begin{bmatrix}
        23 & 34 \\
        31 & 46
        \end{bmatrix}
        $$
        

- 행렬은 분배법칙(Distributive)이 성립한다.

$$
A(B + C) = AB + AC
$$

- 행렬은 결합법칙(Associative)이 성립한다.

$$
A(BC) = (AB)C
$$

- 전치행렬(Transpose Matrix) 성질

$$
(AB)^T = B^TA^T
$$

- 역행렬(Inverse Matrix) 성질

$$
(AB)^{-1} = B^{-1}A^{-1}
$$

## 행렬의 내적
---------

numpy의 `np.inner`는 $i$**번째 행 벡터와** $j$**번째 행 벡터 사이의 내적**을 성분으로 가지는 행렬을 계산한다. 행렬 곱셈과는 다른 연산을 수행한다. 

$$
\mathbf{X}\mathbf{Y}^T = \bigg( \sum_{k} x_{ik} \ y_{jk} \bigg)
$$

주의할 점은 수학에서 말하는 내적과는 다르다. 수학에서는 보통 $tr(\mathbf{X}\mathbf{Y}^T)$, 즉 $\mathbf{X}\mathbf{Y}^T$의 대각 원소를 모두 더한 **스칼라**를 행렬의 내적으로 정의한다. `np.inner`는 trace를 하지 않고 $\mathbf{X}\mathbf{Y}^T$까지만 계산하므로 결과가 **행렬**이다. 즉 `np.inner(X, Y)`는 `X @ Y.T`와 같다.

| 연산 | numpy | 계산하는 것 | 모양 조건 |
|---|---|---|---|
| 행렬곱 | `X @ Y` (`np.matmul`) | $\mathbf{X}$의 행벡터 · $\mathbf{Y}$의 **열벡터** | $\mathbf{X}$의 열 수 = $\mathbf{Y}$의 행 수 |
| 행렬 내적 | `np.inner(X, Y)` | $\mathbf{X}$의 행벡터 · $\mathbf{Y}$의 **행벡터** | $\mathbf{X}$의 열 수 = $\mathbf{Y}$의 열 수 |


## 행렬곱의 활용
-----------

행렬을 이해하는 방법 중 또 다른 하나는 `벡터공간에서 사용되는 연산자(operator)`로 이해할 수 있다.

> 연산자 : 함수를 벡터 공간에서 사용하는 경우

$$
z_\textcolor{red}{i} = \sum_\textcolor{blue}{j} a_{\textcolor{red}{i}\textcolor{blue}{j}} \ x_\textcolor{blue}{j} \\ 
$$

$$
\begin{bmatrix}
z_1 \\
z_2 \\
\vdots \\
z_n
\end{bmatrix} = 
\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1m} \\
a_{21} & a_{22} & \cdots & a_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
a_{n1} & a_{n2} & \cdots & a_{nm}
\end{bmatrix} =
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{bmatrix}
$$

벡터 $\mathbf{x}$를 벡터 $\mathbf{z}$ 로 보내주는 함수와 같은 역할을 하는 행렬 $\mathbf{A}$를 연산자이며, 다른 말로 'm차원 공간에 있는 벡터를 n차원 공간의 벡터로 보내줄 때 연산자를 사용한다.' 라고 말할 수 있다.

즉, 행렬곱을 통해 벡터를 `다른 차원의 공간`으로 보낼 수 있다. 

행렬곱을 통해서 `패턴을 추출`하고 `데이터를 압축`할 수 있으며, **모든 선형변환(linear transform)은 행렬곱으로 계산**할 수 있다.


## 역행렬
----------

어떤 행렬 $\mathbf{A}$ 의 연산을 거꾸로 되돌리는 행렬을 `역행렬(inverse matrix)`이라 부르고 $\mathbf{A}^{-1}$ 라 표기한다. 

역행렬은 **행과 열 숫자가 같고 행렬식(determinant)이 아닌 경우**에만 계산할 수 있다.

역행렬 성질은 아래 수식과 같이 표현할 수 있다.

$$
\mathbf{A}\mathbf{A}^{-1} = \mathbf{A}^{-1}\mathbf{A} = \mathbf{I}
$$

역행렬은 `numpy.linalg.inv`를 이용하면 구할 수 있다.

### 연립방정식

`np.linalg.inv`를 이용하면 연립방정식의 해를 구할 수 있다.

$$
\begin{align*}
a_{11}x_1 + a_{12}x_2 + &\cdots + a_{1m}x_m = b_1 \\
a_{21}x_1 + a_{22}x_2 + &\cdots + a_{2m}x_m = b_2 \\
&\vdots \\
a_{n1}x_1 + a_{n2}x_2 + &\cdots + a_{nm}x_m = b_n \\
\end{align*}
$$

이를 행렬과 벡터를 이용해서 표현하면 다음과 같이 표현할 수 있다.

$$
\mathbf{A} \mathbf{x} = \mathbf{b}
$$

만약, $n = m$이고 $\mathbf{A}$ 의 행렬식이 0이면 역행렬을 이용해서 연립방정식의 해를 구할 수 있다.

$$
\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}
$$