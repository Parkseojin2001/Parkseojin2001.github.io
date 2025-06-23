---
title: "선형대수학의 기초"
description: "선형대수의 기초 개념에 대한 포스트 (From 네이버 부트코스트의 인공지능을 위한 선형대수 강의)"

categories: [Math for AI, Linear Algebra]
tags: [linear-algebra]

permalink: /math/linear-algebra_1/

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-06-23
last_modified_at: 2025-06-23
---

## 스칼라, 벡터 그리고 행렬
---------

### 스칼라(Scalar)

**스칼라(Scalar)**는 방향을 갖지 않고 **크기**만 갖는 개념으로 단일 숫자를 의미한다. 

$$s \in \mathbb{R}$$

e.g.,
- $a = 3$
- $x = -1.2$
- $\theta = 90^\circ$

### 벡터(Vector)

**벡터(Vector)**는 **수직 또는 수평으로 정렬된 숫자들의 집합**이며, 크기와 방향을 가진다.

$$
\mathbf{x} = \begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{bmatrix}
$$

벡터는 순서가 있는 수의 리스트라고 할 수 있으며, 순서가 다르면 같은 벡터가 아니다.

$$
\begin{bmatrix}
2 \\
1 \\
0
\end{bmatrix} \neq
\begin{bmatrix}
1 \\
2 \\
0
\end{bmatrix}
$$


> 순서를 없는 리스트는 집합(set)이다.

### 행렬(Matrix)

**행렬(Matrix)**은 **2차원 숫자 배열**로, 다수의 벡터로 구성된다. 행(row)과 열(column)의 집합이기도 하다.

- 행(Row): 수평적인 벡터
- 열(Column): 수직적인 벡터
- 행렬 크기: 행(Row) $\times$ 열(Column)

$$
A = \begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
$$

## 열 벡터와 행 벡터
-------

### 열 벡터(Column Vector)

**열 벡터(Column Vector)**는 **세로 방향으로 나열된 1차원 배열**로, 보통 이 방식으로 표현한다.

$$
\mathbf{x} = \begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{bmatrix} \in \mathbb{R^n} = \mathbb{R^{n+1}}
$$

### 행 벡터(Row Vector)

**행 벡터(Row Vector)**는 **가로 방향으로 나열된 1차원 배열이다.** 열 벡터을 Transpose한 벡터이기도 하다.

$$
\mathbf{x}^{T} = \begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{bmatrix}^T = \begin{bmatrix}
x_1 & x_2 & \cdots & x_n
\end{bmatrix} \in \mathbb{R}^{1 \times n}
$$

## 행렬 표기법
--------

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

- $A_{ij}$ : 행렬 $A$의 $i$ 번쨰 행, $j$ 번째 열의 요소
- $A_{i,:}$ : 행렬 $A$의 $i$ 번째 형 벡터
- $A_{:,j}$ : 행렬 $A$의 $j$ 번째 열 벡터
    
## 벡터와 행렬의 연산
-----

### 벡터 / 행렬 덧셈과 곱셈

- 벡터와 행렬의 덧셈은 같은 위치에 있는 요소끼리 더한다.
    - 행렬(or 벡터)의 크기가 모두 동일해야한다. ($A, B, C \in \mathbb{R}^{m \times n}$)
    - 행렬(or 벡터)의 뺄셈도 동일함

- 벡터와 행렬의 스칼라 곱셈은 각각의 모든 요소에 스칼라 값을 곱한다.
    - e.g.,
      $$ 
        c \begin{bmatrix}
        x_1 \\
        x_2 \\
        \vdots \\
        x_n
        \end{bmatrix} = 
        \begin{bmatrix}
        cx_1 \\
        cx_2 \\
        \vdots \\
        cx_n
        \end{bmatrix}, 
        c\begin{bmatrix}
        w & x \\
        y & z
        \end{bmatrix} =
        \begin{bmatrix}
        cw & cx \\
        cy & cz
        \end{bmatrix}
        $$

- 행렬과 행렬끼리의 곱셈: $C_{ij} = \sum_{k}A_{i, k}B_{k, j}$
    - 곱하려는 행렬의 첫번째 행렬의 열의 개수와 두번째 행렬의 행의 개수는 동일해야 한다.
    - e.g.,
    $$
     \begin{bmatrix}
        1 & 6 \\
        3 & 4 \\
        5 & 2
        \end{bmatrix}
        \begin{bmatrix}
        1 & -1 \\
        2 & 1
        \end{bmatrix} = 
         \begin{bmatrix}
        13 & 5 \\
        11 & 1 \\
        9 & -3
        \end{bmatrix}
        $$

### 행렬의 성질

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
A(B + C) = AB + AB
$$

- 행렬은 결합법칙(Associative)이 성립힌다.

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