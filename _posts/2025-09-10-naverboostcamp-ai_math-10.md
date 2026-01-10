---
title: "[BoostCamp AI Tech / AI Math] 통계학(1)"
description: "모수의 개념과 모수를 추정하는 방법인 최대가능도 추정법과 정규분포, 카테고리분포에서의 최대가능도 추정법에 대한 포스트입니다."

categories: [NAVER BoostCamp AI Tech, AI Math]
tags: [Statistics]

permalink: /naver-boostcamp/statistics/02

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-09-10
last_modified_at: 2025-09-10
---

## 통계학
-----------

**통계적 모델링은 적절한 가정 위에서 확률분포를 추정(inference)**하는 기법이며, 이는 기계학습과 통계학이 공통적으로 추구하는 목표이다.

이때, 데이터가 특정 확률분포를 따른다고 선험적으로 가정한 후 그 분포를 추정하는 방법을 `모수적(parametric) 방법론`이라고 한다.

<img src="../assets/img/post/naver-boostcamp/parameter.png">

데이터가 아래 특징을 가지면 특정 확률분포로 모델링할 수 있다.

- 데이터가 2개의 값(0 또는 1)만 가지는 경우 &rarr; 베르누이 분포
- 데이터가 $n$개의 이산적인 값을 가지는 경우 &rarr; 카테고리 분포
- 데이터가 [0, 1] 사이에서 값을 가지는 경우 &rarr; 베타 분포
- 데이터가 0 이상의 값을 가지는 경우 &rarr; 감마 분포, 로그정규 분포 등
- 데이터가 $\mathbb{R}$ 전체에서 값을 가지는 경우 &rarr; 정규 분포, 라플라스 분포 등

확률분포마다 모수(parameter)를 추정하는 방법들이 있으며, 데이터를 통한 검정(testing) 과정을 통해 적절한 분포를 선택했는지 확인해야 한다.

하지만, 유한한 개수의 데이터만 관찰해서 모집단의 분포를 정확하게 알아낸다는 것은 불가능하므로 **근사적으로 확률분포를 추정**할 수 밖에 없다.

반대로 특정 확률분포를 가정하지 않고 데이터에 따라 모델의 구조 및 모수의 개수가 유연하게 바뀌면 `비모수(nonparametric) 방법론`이라 부르며 기계학습에서 배우는 상당수의 방법론이 이에 해당한다.


## 최대가능도 추정
---------

통계적 기계학습에서 가장 많이 사용되는 모델 학습원리 중 하나는 `최대가능도추정법(maximum likelihood estimation, MLE)`이다.

$$
\hat{\theta}_{MLE} = \underset{\theta}{\operatorname{argmax}} \ L(\theta; x) = \underset{\theta}{\operatorname{argmax}} \ P(x | \theta)
$$

데이터가 주어져 있는 경우에 그 데이터를 가지고 해당 모수가 실제로 적절한지 아닌지를 평가할 때 사용한다. 

즉, 가능도(likelihood) 함수 $L(\theta, \mathbf{x})$ 는 모수 $\theta$를 따르는 분포에서 데이터 $x$를 관찰할 가능성을 뜻한다.

이떄, 이 가능성을 가장 극대화시키는 파라미터 $\theta$를 찾는것을 MLE라고 한다.

데이터 집합 $X$가 **독립적으로 추출되었을 경우 로그가능도를 최적화**한다.

$$
L(\theta; X) = \prod_{i=1}^n P(x_i | \theta) => log \ L(\theta; X) = \sum_{i=1}^n log \ P(x_i\theta)
$$

$$
\hat{\theta}_{MLE} = \underset{\theta}{\operatorname{argmax}} \ L(\theta; \mathbf{x})
$$

로그가능도를 사용하는 이유는 아래와 같다.

- 로그가능도를 최적화하는 모수 는 가능도를 최적화하는 MLE가 된다.
- 데이터의 숫자가 적으면 상관없지만 **만일 데이터의 숫자가 수억 단위가 된다면 컴퓨터의 정확도로는 가능도를 계산하는 것은 불가능**하다.
- 데이터가 독립일 경우, 로그를 사용하면 가능도의 곱셈을 로그가능도의 덧셈으로 바꿀 수 있기
때문에 컴퓨터로 연산이 가능해진다.
- 경사하강법으로 가능도를 최적화할 때 미분 연산을 사용하게 되는데, 로그가능도를 사용하면 **연산량을 $O(n^2)$에서 $O(n)$으로 줄여준다**.
- 대게의 손실함수의 경우 경사하강법을 사용하므로 **음의 로그가능도(negative log-likelihood)를 최적화**하게 된다.

### 최대가능도 추정법 예제: 정규분포

정규분포를 따르는 확률변수 $X$로부터 독립적인 표본 {$x_1, \ldots, x_n$} 을 얻었을 때 최대가능도 추정법을 이용하여 모수를 추정하면?

$$
\hat{\theta}_{MLE} = \underset{\theta}{\operatorname{argmax}} \ L(\theta; \mathbf{x}) = \underset{\theta}{\operatorname{argmax}} \ P(\mathbf{x} | \theta)
$$

주어진 모수에 대해서 실제로 정규분포 밀도함수를 쓸 수 있다.

$$
\begin{align*}
log \ L(\theta; \mathbf{X}) = \sum_{i=1}^n log \ P(x_i | \theta) &= \sum_{i=1}^n log \ \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{|x_i-\mu|^2}{2 \sigma^2}} \\
&= - \frac{n}{2} log 2\pi \sigma^2 - \sum_{i=1}^{n} \frac{|x_i-\mu|^2}{2\sigma^2}
\end{align*}
$$

위의 식을 미분하고 미분이 모두 0이 되는 $\mu$, $\sigma$를 찾으면 가능도를 최대화하게 된다.

$\mu$, $\sigma$의 각각의 미분을 구하면 아래와 같다.

$$
\begin{align*}
0 &= \frac{\partial log L}{\partial \mu} = - \sum_{i=1}^n \frac{x_i - \mu}{\sigma^2} \Rightarrow \hat{\mu}_{MLE} = \frac{1}{n} \sum_{i=1}^{n} x_i \\ 
0 &= \frac{\partial log L}{\partial \sigma} = \frac{n}{\sigma} + \frac{1}{\sigma^3} \sum_{i=1}^n |x_i - \mu|^2 \Rightarrow \hat{\sigma}^2_{MLE} = \frac{1}{n} \sum_{i=1}^n (x_i - \mu)^2
\end{align*}
$$

- 가능도를 최대화하는 추정 평균 $\hat{\mu} _{MLE}$ 는 데이터들의 산술평균으로, 추정 분산 $\hat{\sigma}^2 _{MLE}$ 는 $(데이터 - 평균)^2$ 의 산술평균이 된다.
- 이 때 주의할 점은, 표본분산을 구할 떄는 $N-1$로 나누었는데, `MLE`로 구한 분산은 $N$으로 나눈다는 것이다.
    - 즉, **MLE는 불편추정량을 보장하진 않는다.**
    - 그러나 통계에서는 말하는 consistency는 보장한다.


### 최대가능도 추정법 예제: 카테고리 분포

`카테고리분포` Multinoulli($x; p_1, \cdots , p_d$)를 따르는 확률변수 $X$로부터 독립적인 표본 ${x_1, \cdots, x_n}$을 얻었을 때, **`최대가능도추정법(MLE)`**을 이용하여 모수를 추정해보자.

어떤 표본들이 어떤 케이스에 해당하는지에 따라 살아남는 표본들만 활용된다.
- $p_k^{x_i, k}$ 에서 $x_i, k$가 1일 때만 살아남고 0일 땐 1로 곱셈에 아무런 영향을 주지 않는다.

케테고리 분포는 아래와 같은 제약 조건을 만족해야한다.

$$
\sum_{k=1}^d p_k = 1
$$


$$
\hat{\theta}_{MLE} = \underset{p_1, \cdots, p_d}{\operatorname{argmax}} \ log \ P(\mathbf{x_i};\theta) = \underset{p_1, \cdots, p_d}{\operatorname{argmax}}  log \bigg( \prod_{i=1}^n \prod_{k=1}^d p_k^{x_i, k} \bigg)
$$

- 만약 해당 데이터 $x_{i, k}$가 0이 되면, $p_{k}^{x_{i, k}}$는 $p_k^{0}$이 되어 1이 된다.
- 반면 $x_{i, k}$가 1이면, $p_k^{x_{i, k}}$는 $p_k$가 된다.

즉, **실제로는 해당 클래스 $k$ 에 해당하는 확률 $p_k$ 하나만 남는다.**

위의 식을 풀이해보면, 

$$
log \bigg( \prod_{i=1}^n \prod_{k=1}^d p_k^{x_i, k} \bigg) = \sum_{k=1}^{d} \bigg(\sum_{i=1}^{n} x_{i, k} \bigg) \ log \ p_k
$$

- 이 때, $\sum_{i=1}^{n} x_{i, k}$는 $n_k$로 치환할 수 있다.
    - $n_k$는, 주어진 각 데이터 $x_{i, k}$들이 대해서 $k$ 값이 1인 데이터의 개수를 의미한다.

치환하여 다시 식을 정리하면,

$$
log \bigg( \prod_{i=1}^n \prod_{k=1}^d p_k^{x_i, k} \bigg) = \sum_{k=1}^{d} n_k \ log \ p_k \ \ \ \text{with} \ \ \ \sum_{k=1}^{d} p_k = 1
$$

- 오른쪽 제약식을 만족하면서 왼쪽 목적식을 최대화하는 것이 우리가 구하는 `MLE`가 된다.
- 이렇게 목적식에 제약식이 있는 경우에는, 그냥 미분값이 0이 되는 값을 구하는 것이 아니라, `라그랑주 승수법(Lagrange multiplier method)`을 이용하여 목적식을 수정해준다.
    - `라그랑주 승수법`은 최적화 문제를 푸는 데에 사용된다.

$$
=> \mathcal{L} (p_1, \cdots, p_k, \lambda) = \sum_{k=1}^d n_k \ log \ {p_k} + \lambda(1 - \sum_{k} p_k)
$$

- `라그랑주 승수법`을 이용해, 제약식을 양변으로 넘겨준 상태에서 라그랑주 승수에 해당하는 $\lambda$ 를 곱해준 식을 목적식에 더해주어서 새로운 목적식을 만들어 준다.
    - 이 새로운 목적식의 최적화로, 제약식도 만족하면서, $log \ L$을 만족시키는 모수 $p_1, \cdots, p_k$를 구할 수 있다.

$$
\begin{align*}
0 &= \frac{\partial \mathcal{L}}{\partial p_k} = \frac{n_k}{p_k} - \lambda \\
0 &= \frac{\partial \mathcal{L}}{\partial \lambda} = 1 - \sum_{k=1}^{d} p_k
\end{align*}
$$

이를 조합하면,

$$
p_k = \frac{n_k}{\sum_{k=1}^{d} n_k}
$$

- 분모에 해당하는 값은, 데이터 개수 $n$ 과 같다.
- 그러므로, $p_k = \frac{n_k}{n}$
- 즉, `카테고리분포`의 `MLE`는 각각의 class에 해당하는 count 수, 즉 **경우의 수를 세어서 전체 중의 비율을 구하는 것**임을 알 수 있다.




