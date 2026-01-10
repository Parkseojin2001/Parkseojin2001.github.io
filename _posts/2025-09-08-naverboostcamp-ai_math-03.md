---
title: "[BoostCamp AI Tech / AI Math] 행렬식(Determinant)"
description: "행렬식과 계수를 통해 행렬의 성질을 파악하고 유사역행렬을 이용하여 일반적인 연립방정식과 선형회귀 분석 문제를 해결하는 과정을 정리한 포스트입니다."

categories: [NAVER BoostCamp AI Tech, AI Math]
tags: [linear-algebra, determinant, pseudoinverse]

permalink: /naver-boostcamp/linear-algebra/03

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-09-08
last_modified_at: 2025-09-08
---

행렬식(determinant)의 절대값은 행렬을 구성하는 벡터로 만들어낸 `다포체(polytope)의 부피의 크기`와 같다.

$$
\mathbf{A} = \begin{bmatrix}
a & b \\
c & d
\end{bmatrix} 
$$

<img src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEgwvh325LUC35qBOvI6bnCnyKYpzH4l-JtoZMCZkQm1f0U7T5zj5ywwE5OP2LgBDz6eE2DqSY23QOyXeh_Gop70dVG7zg2HDirZJd0BayINeMNiYt3IpUvz-r_YbM0420B900_GYdNLlrek/w1200-h630-p-k-no-nu/500px-Area_parallellogram_as_determinant.svg.png" width="300" height="500">

위의 그림에서 사각형의 면적이 행렬식의 절댓값이다.

$$
det(\mathbf{A}) = ad - bc
$$

이를 확장하여 3차원 같은 경우는 아래처럼 표현할 수 있다.

<img src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEj-GrqK_bLwQ92HhNKnsz8HctkGKdWiZtUeyaJgyK_01ZXzg4bSRGl_4eVpHjVA3YzkXpcRegYXcFC0_q5sIoNEG2msOlQv89oJfPhTNtCxNustFY6f-mFFkKpTlS4BwIL9NfvhqG5R0cwU/s280/500px-Determinant_parallelepiped.svg.png">

만약 행렬을 구성하는 벡터들이 서로 **평행하다면** 평행사변형을 만들어내지 못하므로 행렬식이 0이며 이 행렬을 **역행렬이 없는** 행렬이다.

이를 n차원으로 확장하면, 행렬을 구성하는 벡터 일부를 **선형결합해서 다른 벡터를 만들어 낼 수 있다면** 다포체를 만들어내는게 불가능하므로 행렬식이 0이다. 또한, **역행렬이 없는** 행렬임을 알 수 있다.


이를 정리하면 다음과 같이 말할 수 있다.

행렬식은 어떤 행렬을 구성하는 점들이 서로 선형 결합해서 만들어질 수 없는 즉, `선형독립(linearly independent)`인 경우에만 역행렬을 계산할 수 있다.

이를 다른 말로 하면, 행렬 안에 속하는 벡터 중 **선형 독립인 벡터의 개수(rank)가 전체 행렬을 이루는 벡터의 개수가 같을 때** 역행렬을 구할 수 있으며 행렬식이 0이 아니다.

> `rank` : 행렬을 구성하는 벡터 중 선형독립인 벡터의 개수 <br>

$$
det(\mathbf{A}) \ne 0 \iff rank(\mathbf{A}) = n
$$

## 일반적인 연립방정식
-----------

만약, $n \ne m$ 인 경우 역행렬을 계산할 수 없다. 그렇다면 이때는 어떻게 구할 수 있을까?

이러한 상황에서 연립방정식의 해를 구하는 경우는 두 가지로 나눌 수 있다.

- 해가 무수히 많은 경우(부정)
    
    $$
    \begin{align*}
    &\mathbf{x}_1 \in {\mathbf{x} : \mathbf{Ax} = \mathbf{b}} \ \text{해집합} \\
    &\mathbf{x}_0 \in {\mathbf{x} \ne 0 : \mathbf{Ax} = 0} \ \text{영공간(0제외)} \\
    \Rightarrow \ &\mathbf{A}(\mathbf{x}_1 + \alpha \mathbf{x}_0) = \mathbf{A}\mathbf{x}_1 + \alpha \mathbf{A}\mathbf{x}_0 = \mathbf{b} \\
    \end{align*}
    $$

    이러한 경우 $\mathbf{x}_1$ 뿐만 아니라 $\alpha\mathbf{x}_0$ 도 해집합에 속하기 때문에 해가 유일하지 않다. 

    - 만약 영공간이 0밖에 없을 땐 해가 무수히 많은 경우에 속하지 않는다.

- 해가 없는 경우(불능)

    $$
    \begin{align*}
    {}^{\exists}\!\ \mathbf{b} \notin \{ \mathbf{Ax} : \mathbf{x} \in \mathbb{R}^m \}
    \end{align*}
    $$

이렇게 역행렬을 계산할 수 없다면 `유사역행렬(pseudo-inverse)` 또는 `무어-펜로즈(Moore-Penrose) 역행렬` $\mathbf{A}^{+}$ 을 이용한다.

여기서 행렬 A는 $n \times m$인 행렬이다.

- $n \ge m$ 인 경우 : $\mathbf{A}^{+}= (\mathbf{A}^T \mathbf{A})^{-1} \mathbf{A}^{T}$
- $n \le m$ 인 경우 : $\mathbf{A}^{+}= \mathbf{A}^{T}(\mathbf{A}\mathbf{A}^{T})^{-1} $

유의할 점은 행과 열의 개수가 다를 때 **둘 중 작은 값을 rank 값으로 가져야만** 위의 공식을 사용할 수 있다. 

위의 과정을 코드로 구현하면 다음과 같다.

- 유사역행렬은 `numpy.linalg.pinv`로 구할 수 있다.

```python
import numpy as np

A = np.array([[0, 1],
              [1, -1],
              [-2, 1]])
print("Rank of A: ", np.linalg.matrix_rank(A))  # Rank of A: 2

# 유사역행렬
np.linalg.inv(A.transpose() @ A) @ A.transpose()
np.linalg.pinv(A)

# 항등행렬
np.linalg.pinv(A) @ A
```

## 선형회귀 분석
-----------

`np.linalg.pinv`를 이용하면 데이터를 선형모델(linear model)로 해석하는 `선형회귀식`을 찾을 수 있다.

선형회귀분석은 $\mathbf{X}$ 와 $\mathbf{y}$ 가 주어진 상황에서 계수 $\beta$를 찾아야 한다.

이를 수식으로 표현하면 

$$
\mathbf{X}\beta = \mathbf{y}
$$

이를 좀 더 풀어서 쓰면 아래와 같다.

<img src="https://images.velog.io/images/recoder/post/09f7929a-7561-4e27-aaf7-351c2b056e50/image.png" height="200" width="350">

선형회귀는 데이터가 변수의 개수보다 많은 경우이며, 이는 열($m$)의 개수보다 행($n$)의 개수가 많다.

<img src="https://images.velog.io/images/recoder/post/a9c7e0e7-0d97-4831-87e2-99fc671172b1/image.png" height="200" width="350">

이는 식의 개수가 변수의 개수보다 더 많은 경우이므로 불능형 연립방정식이므로 방정식을 푸는 것은 불가능하다.

그러므로 최대한 $\mathbf{y}$에 근사할 수 있도록 $\mathbf{X}$ 에서 $\beta$ 벡터를 찾아야한다.

이를 수식으로 표현하면 다음과 같다.

$$
\begin{align*}
\mathbf{X}\beta &= \hat{\mathbf{y}} \approx \mathbf{y} \\ 
\Rightarrow \beta &= \mathbf{X}^{+} \mathbf{y}\\
&= (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
\end{align*}
$$

위의 식은 $\underset{\beta}{min} \lVert \mathbf{y} - \mathbf{\hat{y}} \rVert _2$ 로 $L_2$-norm이라고 말할 수 있으며 이 값을 최소화 해야한다.

이를 최소화하는 $\beta$를 찾기 위해서는 `Moore-Penrose 역행렬`을 이용하면 찾을 수 있다.

이를 코드로 작성하면 다음과 같다.

```python
# Scikit Learn 활용
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
y_test = model.predict(x_test)

# Moore-Penrose 역행렬
X_ = np.array([np.append(x, [1]) for x in X])   # intercept 항 추가
beta = np.linalg.pinv(X_) @ y
y_test = np.append(x, [1]) @ beta
```



