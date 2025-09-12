---
title: "행렬 분해"
description: "행렬 분해 중 고유값 분해와 특이값 분해 대해 그리고 고유값 분해를 활용한 주성분 분석(PCA)에 대한 내용 정리 포스트입니다."

categories: [Math for AI, Linear Algebra]
tags: [linear-algebra, eigen decomposition, PCA, singular value decomposition]

permalink: /naver-boostcamp/linear-algebra/04

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-09-08
last_modified_at: 2025-09-08
---

행렬분해(matrix decomposition)는 행렬을 **여러 행렬들의 곱으로 분해해서 표현**하는 기법을 말한다.

$$
\mathbf{A} = \mathbf{A}_1\mathbf{A}_2 \cdots \mathbf{A}_m
$$

행렬 분해는 여러 종류가 있고 목적에 따라 쓰이는 방법이 다르다.

- 고유값(eigenvalue) 분해
- 특이값(singular value) 분해
- LU(lower-upper) 분해, OR 분해
- 비음수행렬분해(NMF)

## 고유값 분해
---------

행렬에 어떤 벡터 $\mathbf{v}$를 곱했을 때 그 벡터의 상수배가 되는 경우, 이 벡터를 `고유벡터(eigenvector)`라 부르고 상수를 `고유값(eigenvalue)`이라 부른다.

이때 행렬의 행과 열이 동일할 때, 고유벡터와 고유값을 구할 수 있다.

$$
\mathbf{A}\mathbf{v} = \lambda \mathbf{v}
$$

고유값과 고유벡터가 $n$개 있을 때 아래와 같은 관계식이 성립힌다.

$$
\mathbf{A} \begin{bmatrix}
| & | & | \\
\mathbf{v}_1 & \cdots & \mathbf{v}_n \\
| & | & | \\
\end{bmatrix} = \begin{bmatrix}
| & | & | \\
\lambda_1 \mathbf{v}_1 & \cdots & \lambda_n \mathbf{v}_n \\
| & | & | \\
\end{bmatrix} = \begin{bmatrix}
| & | & | \\
\mathbf{v}_1 & \cdots & \mathbf{v}_n \\
| & | & | \\
\end{bmatrix} \begin{bmatrix}
\lambda_1 & &\\
& \ddots & \\
& & \lambda_n
\end{bmatrix}
$$

위의 수식을 간략히 나타내면

$$
\mathbf{A}\mathbf{V} = \mathbf{V}\mathbf{\Lambda}
$$

만약 모든 고유벡터들끼리 선형독립이면 $\mathbf{V}$ 의 역행렬이 존재한다. 이를 이용해서 행렬 $\mathbf{A}$ 를 아래와 같이 나타낼 수 있다.

$$
\mathbf{A} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^{-1}
$$

이를 `eigenvalue decomposition` 이라고 한다.

### 주성분 분석(PCA)

대표적인 예시로는 `PCA`가 있다.

PCA는 데이터를 저차원공간으로 효율적으로 압축하기 위해 `고유값분해`를 사용한다.


데이터를 저차원으로 축소하려면 어떤 기준으로 압축 즉, 어떤 $\mathbf{v}$ 를 선택해야 가장 정보손실이 덜할지를 구해야한다.

사용하는 방법은 데이터(x, y)를 벡터 $\mathbf{v}$ 로 정사영시키는 것이다.

이를 수식으로 나타내면 다음과 같다.

$$
Proj_{\mathbf{v}}(\mathbf{x}_i) = <\mathbf{x}_i, \mathbf{v}> \mathbf{v}
$$

$$
\underset{\mathbf{v}}{\text{minimize}} \sum_i \lVert \mathbf{x}_i - Proj_\mathbf{v} (\mathbf{x}_i) \rVert = \sum_i (\lVert \mathbf{x}_i \rVert^2 - 2<\mathbf{x}_i, \mathbf{v}>^2 + <\mathbf{x}_i, \mathbf{v}>^2 \lVert \mathbf{v} \rVert^2)
$$

$$
\text{subject to} \ \lVert \mathbf{v} \rVert = 1
$$

$\lVert \mathbf{v} \rVert = 1$ 이기 때문에 이를 대입해서 다시 수식을 쓰면 아래와 같이 쓸 수 있다.

$$
\underset{\mathbf{v}}{\text{minimize}} \sum_i \lVert \mathbf{x}_i - Proj_\mathbf{v} (\mathbf{x}_i) \rVert = \sum_i (\lVert \mathbf{x}_i \rVert^2 - <\mathbf{x}_i, \mathbf{v}>^2)
$$

이 때, $\mathbf{v}$ 만 고려해서 최적화를 진행하기 때문에 $\mathbf{x}_i$는 신경쓰지 않아도 된다.

최종적인 식은 

$$
\underset{\mathbf{v}}{\text{maximize}} \sum_i <\mathbf{x}_i, \mathbf{v}>^2 = \lVert \mathbf{X}^T \mathbf{v}\rVert^2 = \mathbf{v}^T\mathbf{X}\mathbf{X}^T\mathbf{v}
$$

이 목적식을 통계학적으로 분석을 해보면 **정사영된 데이터의 분산을 가장 크게 만드는 단위 벡터** $\mathbf{v}$를 찾는 문제라는 것을 알 수 있다.

주로 이런 문제를 해결하기 위해 사용하는 방법은 `라그랑주 승수법`이다.

$$
\nabla_{\mathbf{v}} \lVert \mathbf{X}^T \mathbf{v} \rVert ^2 = \nabla_{\mathbf{v}} \lambda (\lVert \mathbf{v} \rVert ^2 - 1)
$$

$$
\mathbf{X}\mathbf{X}^T \mathbf{v} = \lambda \mathbf{v}
$$

위의 수식을 다시 최적화 식에 대입해보면 다음과 같은 결과가 나온다.

$$
\underset{\mathbf{v}}{\text{maximize}} \sum_i <\mathbf{x}_i, \mathbf{v}>^2 = = \mathbf{v}^T\mathbf{X}\mathbf{X}^T\mathbf{v} = \lambda \lVert \mathbf{v}^2 \rVert = \lambda
$$

이는 정사영했을 때 가장 효율적인 정보 압축이 가능한 후보들은 모두 고유벡터가 된다는 것이다.

이 고유값들 중 가장 큰 값을 `주성분(Principle component)`라고 부른다.

## 특이값 분해
-----------

고유값분해는 행렬이 정사각(square)일 때만 사용 가능하지만 특이값분해(SVD)는 일반적인 행렬에 사용할 수 있는 방법이다.

수식은 다음과 같다.

$$
\mathbf{A} = \mathbf{U}\sum\mathbf{V}^T
$$

- $\mathbf{U}$, $\mathbf{V}$ : orthogonal 행렬
- $\sum$ : 특이값으로 이루어진 대각행렬

SVD를 통해 다음과 같이 분해하는 것이 가능하다.

$$
\mathbf{A} = \sum_{k=1}^{rank(\mathbf{A})} \sigma_k \mathbf{u}_k \mathbf{v}_k^T
$$

이 때 $\sigma_k$는 특이값이다.

특이값 분해를 통해서도 선형회귀분석을 수행할 수 있다.

$$
\begin{align*}
&\mathbf{X}\beta = \hat{\mathbf{y}} \approx \mathbf{y} \\
& \Rightarrow \beta = \mathbf{X}^{+}\mathbf{y} \\
&= \mathbf{V}\sum^{+}\mathbf{U}^T\mathbf{y}
\end{align*} 
$$

SVD를 이용하는 것이 역행렬 계산보다 계산복잡도가 낮다.