---
title: "경사하강법"
description: "미분의 개념과 미분을 이용한 경사하강법에 대한 정리 포스트입니다."

categories: [Math for AI, Gradient Descent]
tags: [Gradient Descent, calculus, differentiation]

permalink: /naver-boostcamp/gradient-descent/01

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-09-09
last_modified_at: 2025-09-09
---

## 미분
----------

미분(differentiation)은 **변수의 움직임에 따른 함수값의 변화를 측정하기 위한 도구**로 최적화에서 제일 많이 사용하는 기법이다.

$$
f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
$$

`sympy.diff`를 가지고 미분을 계산할 수 있다.

```python
import sympy as sym
from sympy.abc import x

sym.diff(sym.poly(x**2 + 2*x + 3), x)
# Poly(2*x + 2, x, domain='ZZ')
```

미분을 시각적으로 분석하면 함수 $f$의 주어진 점 $(x, f(x))$에서의 `접선의 기울기`이다.

<img src="https://ballpen.blog/wp-content/uploads/2022/10/Picture1.jpg" width="500" height="300">

이렇게 한 점에서 접선의 기울기를 알면 어느 방향으로 점을 움직여야 함수값이 `증가`하는지 / `감소`하는지 알 수 있다.

- 증가시키고 싶으면 미분값을 더한다. 
    - 미분값이 음수라서 $x + f'(x) < x$는 **왼쪽으로 이동**하여 함수값이 증가
    - 미분값이 양수라서 $x + f'(x) > x$는 **오른쪽으로 이동**하여 함수값이 증가
- 감소시키고 싶으면 미분값을 빼면 된다.
    - 미분값이 음수라서 $x - f'(x) > x$는 **오른쪽으로 이동**하여 함수값이 감소
    - 미분값이 양수라서 $x - f'(x) < x$는 **왼쪽으로 이동**하여 함수값이 감소

이를 정리하면 다음과 같다.

- **미분값을 더하면 경사상승법(gradient ascent)**이라 하며 함수의 `극댓값`의 위치를 구할 때 사용한다.
- **미분값을 빼면 경사하강법(gradient descent)**이라 하며 함수의 `극솟값`의 위치를 구할 때 사용한다.

<img src="https://bsm8734.github.io/assets/img/sources/2021-01-27-01-19-43.png">

이때, **극값에선 미분값이 0**이되면 경사상승 / 경사하강 방법이 멈춘다.

## 경사하강법
---------

경사하강법의 알고리즘은 아래와 같이 작성할 수 있다.

```python
def func(val):
    fun = sym.poly(x**2 + 2*x + 3)
    return fun.subs(x, val), fun

def func_gradient(fun, val):
    _, function = fun(val)
    diff = sym.diff(function, x)
    return diff.subs(x, val), diff
# gradient: 미분을 계산하는 함수
# init: 시작점  lr: 학습률  eps: 알고리즘 종료조건

def gradient_descent(fun, init, lr=1e-2, eps=1e-5):
    cnt = 0
    val = init
    diff, _ = func_gradient(fun, init)
    while np.abs(diff) > eps:
        val = val - lr * diff
        diff, _ = func_gradient(fun, val)
        cnt += 1
```

> 경사하강법에서 변수 `val`이 벡터인 경우에 경사하강법 알고리즘은 그대로 적용되나 벡터는 절대값 대신 `norm`을 계산하여 종료조건을 설정하여야 한다.