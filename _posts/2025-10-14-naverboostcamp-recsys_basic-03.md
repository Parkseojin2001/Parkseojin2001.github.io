---
title: "변분추론"
description: "변분추론의 개념 및 유도과정과 변분 추론 중, MFVI(Mean-Field Variational Inference)의 개념에 대한 내용을 정리한 포스트입니다."

categories: [Deep Learning, RecSys]
tags: [RecSys, VI, MFVI]

permalink: /naver-boostcamp/recsys/03

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-10-15
last_modified_at: 2025-10-15
---

`변분추론(Variational Inference)`는 생성 모델을 학습하기 위해서는 별도의 과정이다.

주어진 데이터가 있을 때, 이 데이터를 가장 잘 설명할 수 있는 모델 파라미터를 추론하는 방식으로 학습을 하는데 이를 수행하기 어렵기 때문에 근사하는 방식으로 해결한다.


$$
\begin{align*}
&\text{The model : } p_{\theta}(x) \\
&\text{The data : } D = \{ x_1, \ldots, x_n \} \\
&\text{Maximum likelihood : } \theta \leftarrow \text{argmax}_{\theta} \frac{1}{N} \sum_{i} log \ p_{\theta} (x_i) \\
&\text{Maximum likelihood : } \theta \leftarrow \text{argmax}_{\theta} \frac{1}{N} \sum_{i} log \ \int p_{\theta} (x_i, z) \ dz \\
&\text{Alternative : } \theta \leftarrow \text{argmax}_{\theta} \frac{1}{N} \sum_{i} {\color{red}{E_{z \sim p(z | x_i)}}} log \ p_{\theta} (x_i, z)
\end{align*}
$$

이때, 모든 $z$가 아닌 z를 **랜덤하게 posterior에서 샘플링**을 하여 계산한다. 이러한 과정을 여러번 수행한 후 이 수행한 값에 평균을 취한다. 이러한 방식을 `Monte Carlo Approximation` 이라고 한다.

여기서 가질 수 있는 의문점은 왜 **Posterior에서 샘플링해야하는지** 그리고 **어떻게 하면 Posterior를 구할 수 있는지** 이다.

이제부터 이 두 질문의 답을 살펴볼 것이다.

## VI를 위한 기본 개념
-----

위의 두 질문을 답을 이해하기 위해서 `Convex function` 과 `Jensen's Inequality`를 알아보자.
 
`Convex function`와 `Concave function` 은 각각 다음과 같은 조건을 만족한다.

- Convex function

    $$
    \begin{align*}
    &\text{For all } 0 \le t \le 1 \text{ and all } x_1, x_2 \in X \\
    &f(tx_1 + (1 - t)x_2) \le tf(x_1) + (1-t)f(x_2)
    \end{align*}
    $$

    - example: $x, x^2, e^{x}$

- Concave function

    $$
    \begin{align*}
    &f \ : \ X \rightarrow R \\
    &f \text{ is concave if} - f \text{ is convex}
    \end{align*}
    $$

    - example: $log \ x$

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSlITJoZdw5i23yVd2iUR-XH2XEyEIqX326Gg&s">

위 두 함수에 대해 `Jensen's Inequality`를 만족한다.

$$
\begin{align*}
&\phi(E[X]) \le E[\phi(X)] \text{ where } \phi \text{ is convex function} \\
&\phi(E[X]) \ge E[\phi(X)] \text{ where } \phi \text{ is concave function} \\
\end{align*}
$$

이와 관련된 예로는 다음과 같은 수식이 있다.

$$
Var(x) = E[x^2] - E[x]^2 \ge \Rightarrow E[x]^2 \le E[x^2]
$$

## Variational Inference
----------
