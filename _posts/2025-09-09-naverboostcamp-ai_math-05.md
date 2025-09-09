---
title: "텐서(Tensor)"
description: "텐서 구조와 텐서 연산을 위한 도구 einsum과 einops의 사용법에 대한 정리 포스트입니다."

categories: [Math for AI, Linear Algebra]
tags: [linear-algebra, pytorch, Tensor, einsum, einops]

permalink: /naver-boostcamp/linear-algebra/05

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-09-09
last_modified_at: 2025-09-09
---

N차원 텐서(tensor)는 `N-1차원 텐서`를 원소로 가지는 `배열`이며 N차원 텐서는 **인덱스 개수가 N개**가 된다.

벡터는 1차원 텐서라고 할 수 있으며, 행렬은 2차원 텐서라고 말할 수 있다.

텐서를 사용하는 예시로는 영상 데이터가 있으며 이때, 채널 정보를 포함해 3차원 텐서로 표현된다.

이러한 영상 데이터가 여러 개가 있으면 3차원 텐서가 여러 개 모여 4차원 텐서로 이루어진다.

## 텐서 연산
-------

**텐서끼리 같은 모양을 가지면** 덧셈, 뺄셈, 성분곱을 계산할 수 있다.

$$
\begin{align*}
\mathbf{X} \pm \mathbf{Y} &= (x_{i_1 \ldots i_N} \pm y_{i_1 \ldots i_N}) \\
\mathbf{X} \odot \mathbf{Y} &= (x_{i_1 \ldots i_N} \times y_{i_1 \ldots i_N}) \\
\end{align*}
$$

텐서의 곱셈은 Numpy 연산마다 다르게 정의되기 때문에 주의가 필요하다.

$$
\begin{align*}
\text{dot}(\mathbf{X}, \mathbf{Y}) &= (z_{bipj}) = \bigg( \sum_k x_{bik}y_{pkj} \bigg) \\
\text{matmul}(\mathbf{X}, \mathbf{Y}) &= (z_{bij}) = \bigg( \sum_k x_{bik}y_{bkj}\bigg)
\end{align*}
$$

행렬에선 `np.dot`와 `np.matmul`이 같은 기능이지만 3차원 이상의 텐서부터는 다르게 동작할 수 있다.

## einsum 
-----------

`einsum`은 아인슈타인 표기법에서 유래한 것으로 텐서를 활용한 여러 종류의 곱연산에서 편리함을 가진다.

$$
\mathbf{X}\mathbf{Y} = \bigg( \sum_k x_{ik}y_{kj}\bigg)
$$

이를 아인슈타인 표기법으로 표현하면 다음과 같다.

$$
\mathbf{X}\mathbf{Y} = x_{ik}y_{kj} = (z_{ij})
$$

이를 코드로 표현하면 다음과 같다.

```python
# 행렬곱
np.einsum('ik, kj -> ij', X, Y)
```

행렬곱 뿐만 아니라 다른 연산에도 표현할 수 있다.

$$
\begin{align*}
\mathbf{X}^T &= (x_{ji}) \\
\mathbf{X}\mathbf{Y}^T &= (x_{ik}y_{jk}) = (z_{ij})
\end{align*}
$$

이를 코드로 표현하면 다음과 같이 표현할 수 있다.

```python
# 전치행렬
np.einsum('ij ->ji', X)

# 행렬 내적
np.einsum('ik, jk -> ij', X, Y)
```

위의 `dot`와 `matmul`에 einsum을 적용하면 아래와 같다.

```python
# dot 계산
np.einsum('bik, pkj -> bipj', X, Y)

# matmul 계산
np.einsum('bik, bkj -> bij', X, Y)
```

## einops
------------

`einsum`과 더불어 `einops`도 같이 이용하면 텐서의 계산을 다루는 것이 직관적으로 가능하다.

ex. 3차원 텐서의 각 성분 별로 `trace`값을 계산

> trace : 행렬에서 각 대각성분의 합을 계산

$$
Tr(X) = [Tr(X_1), \ldots, Tr(X_d)] = \sum_i x_{ij}
$$

실제 코드는 다음과 같이 작성할 수 있다.

```python
import numpy as np

x = np.array([
    [[1, 2], [3, 4]],
    [[-1, -2], [-3, -4]],
])
np.trace(x) # 원래 정답 : array([5, -5])
# array([-2, -2]) 
```

trace 계산을 einsum으로 표기하면 아래와 같다.

$$
Tr(X) = [(\mathbf{X_b})_{ii}] = (z_b)
$$

이를 코드로 구현하면 아래와 같다.

```python
import numpy as np

x = np.array([
    [[1, 2], [3, 4]],
    [[-1, -2], [-3, -4]],
])
np.einsum('bii -> b', x) # array([5, -5])
```

ex. 텐서들을 2차원으로 변환 후 행벡터끼리 내적을 계산(transformer에서 자주 사용)

```python
X_re = einops.rearrange(X, 'b i j k -> b (i j k)')
Y_re = einops.rearrange(Y, 'b i j k -> b (i j k)')

np.einsum('bi, bi -> b', X_re, Y_re)
```
`np.reshape` 대신 `einops.rearrange`를 쓰면 텐서의 shape를 구체적으로 몰라도 변환 가능하다.

ex. 4차원 (b i j c) 텐서를 3차원 (b k c) 텐서로 변환 후, k 인덱스에 대해 내적 계산을 해서 b 인덱스 별로 평균값 구하기

```python
# 4차원 -> 3차원 변환
X_re = einops.rearrange(X, 'b i j c -> b (i j) c')
X_re = einops.rearrange(X, 'b i j c -> b (i j) c')

# 내적
z = np.einsum('bkc, bkc -> bc', X_re, Y_re, dtype=float)

# b 인덱스 별로 평균값
z_mean = einops.reduce(z, 'b c -> b', 'mean')
```

> `einops.reduce`을 사용할 땐 `data type`을 명시하는 게 좋다.

