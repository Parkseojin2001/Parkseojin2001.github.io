---
title: "딥러닝 기본 - Generative Model"
description: "네이버 부스트코스의 Pre-course 강의를 기반으로 작성한 포스트입니다."

categories: [Naver-Boostcamp, Pre-Course 2]
tags: [Naver-Boostcamp, Pre-Course, pytorch, Generative-Model]

permalink: /boostcamp/pre-course/generative-model/

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-08-07
last_modified_at: 2025-08-07
---

우리는 `생성 모델(Generative Model)`을 생각하면 대부분 `적대적 생성모델(GAN)`을 떠올린다. 그러나 그럴듯한 이미지나 문장을 만드는 것이 생성 모델의 전부는 아니다.

## 생성 모델의 학습 방법
-----------

개(dog)의 이미지가 여러 장 주어졌다고 하자.

생성모델의 확률분포(probability distribution) $p(x)$를 학습함으로써 할 수 있는 것은 다음과 같다.

- `Generation` : 확률분포 $p(x)$와 유사한 어떤 새로운 $x_{new}$를 만들어서 ($x_{new} \sim p(x)$), 개와 유사한 이미지를 만들어낼 수 있다. (sampling)
- `Density estimation` : 어떤 이미지 $x$가 개인지 아닌지 판별해낼 수 있다. $p(x)$가 높게 나온다면 개이고, $p(x)$가 낮게 나온다면 개가 아니다. (anomaly detection)
    - 엄밀한 의미의 생성 모델은 `Discriminator` 모델까지 포함하고 있다. 어떤 이미지가 특정 레이블에 속하는 지 아닌지 판단할 수 있어야 한다.
    - 이처럼 **Discriminator를 포함한 생성 모델을 `명시적 모델(Explicit Model)`이라고 하며, Generation만 할 수 있는 모델을 `암시적 모델(Implicit Model)`이라고 한다.**
- `Unsupervised representation learning` : 이미지들 사이에서 feature(특징,공통점)을 찾아낼 수 있다.(feature learning)

그렇다면 $p(x)$는 무엇일까? $p(x)$는 $x$를 집어넣었을 때 나오는 값일수도 있고, $x$를 샘플링할 수 있는 어떤 모델일수도 있다. 이 $p(x)$를 알기 위해서는 먼저 확률에 대한 간단한 선행지식이 필요하다.

### Basic Discrete Distributions

`베르누이 분포(Bernoulli distribution)` : 앞, 또는 뒤만 나오는 동전 던지기와 같다.(0,1)
- $D$ = [앞면(Head), 뒷면(Tails)]
- 앞면이 나올 확률이 $P(X=Heads)=p$ 라면, 뒷면이 나올 확률은 $P(X=Tails)=1−p$이다.
- 이를 $X∼Ber(p)$로 표현한다.
- 파라미터는 1개($p$ 하나)

`카테고리 분포(Categorical Distribution)` : $m$ 개의 면을 가지는 주사위를 던지는 것과 같다.
- $D$ = [1, 2, ... , m]
- 주사위를 던져 $i$ 가 나올 확률을 $P(Y=i)=p_i$라고 한다면, $\sum_{i=1}^m p_i = 1$이다.
- 이를 $Y∼Cat(p_1, \ldots, p_m)$이라고 한다.
- 파라미터는 $m-1$개

ex. RGB 픽셀 하나의 `결합분포(joint distribution)`를 모델링한다고 생각하자.

RGB 픽셀 하나의 결합분포(joint distribution)를 모델링한다고 생각하자.

- $(r,g,b) \sim p(R,G,B)$
- 가능한 색상의 종류(경우의 수) : $256 \times 256 \times 256$개
- 필요한 파라미터의 개수 : $255 \times 255 \times 255$개

단 하나의 RGB 픽셀을 표현하기 위한 **파라미터 숫자가 굉장히 많다.**

이번엔 $n$개의 binary pixels(하나의 binary image) $X_1, \ldots, X_n$이 있다고 치자.

- 가능한 경우의 수 : $2 \times 2 \times \ldots \times 2 = 2^n$개
- 필요한 파라미터의 개수 : $2^n - 1$개

이처럼, 바이너리 이미지를 나타내는 데에도 **너무 많은 파라미터가 필요**하다.

그러나 파라미터의 개수가 늘어나면 모델의 학습은 일반적으로 잘 되지 않는다. 이 파라미터를 줄일 방법은 없을까?

### Structure Through Independence

위의 binary image 사례에서, 모든 픽셀 $X_1, \ldots, X_n$가 각각 **독립적이라고 가정(independance assumption)**하면, 다음과 같다.

$$
p(x_1, \ldots, x_n) = p(x_1)p(x_2)\cdots p(x_n)
$$

- 가능한 경우의 수 : $2^n$개
- $p(x_1, \ldots, x_n)$을 구하기 위해 필요한 파라미터의 개수 : $n$ 개

동일하게 $2^n$개의 경우의 수를 나타낼 수 있지만, $n$ 개의 파라미터만 있으면 된다.

당연히 말이 안되는 가정이긴 하다. 이미지사진이므로 각 픽셀사이의 분포는 아무래도 인접할수록 비슷할 확률이 높은데, 이를 완전히 무시한 가정이기 때문에, 적당한 분포를 모델링하기에는 좋지 않다.

### Conditional Independence

**기존의 Fully Dependant 모델링과 Independent 모델링의 중간점으로 타협한 것**이 `Conditional Independence`이다.

Conditional Indepence는 다음과 같은 3가지 핵심 룰로 동작한다.

- `연쇄법칙(Chain rule)`

    $$
    p(x_1, \ldots, x_n) = p(x_1)p(x_2|x_1)p(x_3|x_1, x_2)\cdots p(x_n|x_1, \cdots, x_{n-1})
    $$

- `베이즈 정리(Bayes' rule)`

    $$
    p(x|y) = \frac{p(x, y)}{p(y)} = \frac{p(y|x)p(x)}{p(y)}
    $$
​
- `조건부독립`

    $$
    \text{if } x\perp y|z, \text{then} p(x|y,z)=p(x∣z)
    $$

    - $z$가 주어졌을 때 $x$와 $y$가 독립적이라면, $y$, $z$가 주어졌을 때 $x$가 일어날 확률은 그냥 $z$만 주어지더라도 $x$ 가 일어날 확률과 같다. 즉 $y$는 상관이 없다.
    - 이것으로 연쇄법칙의 conditional 부분을 날릴 수 있다.

조건부 독립이 잘 이해되지 않는다면, 적절한 예시를 들어 잘 설명해놓은 다음 글을 참고한다.

참고: [조건부 독립(Conditional Independence)](https://actruce.com/conditional-independence/)

그럼 binary 이미지 예제에서 연쇄법칙을 이용한다고 가정해보자.

필요한 파라미터의 개수는 몇개일까?

- $p(x_1)$ : 1개

- $p(x_2 \| x_1)$ : 2개 (하나는 $p(x_2 \| x_1) = 0$, 나머지 하나는 $p(x_2 \| x_1)=1$)

- $p(x_3 \| x_1, x_4)$ : 4개 (입력 $x_1$, $x_2$를 모두 고려한다)

따라서 , 최종적으로 필요한 파라미터의 개수는 $1+2+2^2+ \cdots +2^{n−1}=2^n−1$로, Fully Dependant 모델과 같다.

이제, `Markov assumption`을 가정해보자. 즉, $X_{i+1}$는 $X_i$에만 dependant하고, $X_1, \ldots, X_i$ 까지에는 independent하다고 가정한다.($X_{i+1}\perp X_1, \ldots, X_{i-1} \| X_i$)

$$
p(x_1, \ldots, x_n) = p(x_1)p(x_2|x_1)p(x_3|x_2)\cdots p(x_n|x_{n-1})
$$

이 경우 파라미터 개수는 $2n−1$이다.

따라서, Markov assumption을 적용시킴으로써 **파라미터의 개수를 지수차원에서 끌어내릴 수 있다.**

`Autoregressive Model`은 이 conditional independance를 잘 활용한 모델이다.

### Auto-regressive Model

28x28크기의 바이너리 이미지(픽셀들의 모음)이 있다고 하자.

우리는 $x \in \{ 0,1 \}^{784}$인 $x$에 대해 $p(x) = p(x_1, \ldots, x_{784})$를 구해야한다. 이 때 $p(x)$를 어떻게 파라미터로 표현할 수 있을까?

- 연쇄법칙을 이용해 결합분포를 나눈다.
- $p(x_{1:784}) = p(x_1)p(x_2 \| x_1)p(x_3 \| x_{1:2}) \cdots$
- 이것을 `AR모델(Autoregressive Model, 자기회귀모델)`이라고 한다.
    - AR모델은 하나의 정보가 이전 정보에 dependant한 것을 의미한다. 바꿔말하면, Markov assumption처럼 직전 정보에만 dependant한것도 AR모델이고, 거꾸로 $x_1, \ldots, x_{i-1}$까지에 모두 dependant한 것도 AR모델이다.
    - **이전의 $n$개 정보들을 고려하는 모델**을 `AR(n) 모델`이라고 부른다.
- 이 때, 주의할 점은 바이너리 픽셀들을 1부터 784까지 순서를 매겼듯이 랜덤한 variable들에 각각 순서를 매겨야 한다는 것이다.
    - 이 순서를 어떻게 정하느냐에 따라 모델의 성능이 달라질수도 있다.

어떤 식으로 Coditional Indepedency를 주는가에 따라 연쇄법칙을 이용해 결합분포를 나누는 방식에 차이가 생기므로, 결과적으로 AR모델의 structure가 달라진다.

## 생성 모델의 종류
---------

### NADE: Neural Autoregressive Density Estimator

<img src="https://blogik.netlify.app/static/8091191a89534c015807b1fd896bcc27/1d69c/nade.png">

$$
p(x_i|x_{1:i-1}) = \sigma(\alpha_i h_i + b_i) \ \text{ where} \ h_i = \sigma(W{<ix_{1:i-1} + c})
$$

$i$번째 픽셀 $x_i$는 그 이전까지의 모든 픽셀들에 대해 dependant하다. 픽셀의 순서가 뒤로 갈수록 받는 입력($x_{1:i-1})의 수가 많아지므로, **weight의 길이가 가변적**이다. 그 이외에는 AR모델과 동일하다.

`NADE`는 `explicit 모델`로, **주어진 입력값의 확률(density)를 계산할 수 있다.**

- 784 픽셀의 바이너리 이미지라고 가정하면, 각 조건부 확률 $p(x_i \| x_{1:i-1})$이 독립적으로 계산될 때 결합분포를 다음과 같이 계산할 수 있다.

$$
p(x_1, \ldots, x_{784}) = p(x_1)p(x_2|x_1) \cdots p(x_{784}|x_{1:783})
$$

- 각각의 $p(x_i\|x_{1:i-1})$을 차례차례 계산해 대입하면, 전체 확률을 알 수 있게 된다.

NADE는 이 확률을 토대로 해당 이미지를 판별하는 **discriminator 역할을 수행할 수 있다.** 그래서 논문 제목에 `Density Estimator`라는 단어가 붙었다.

- "Density Estimator"라는 단어는 explicit model을 표현할 때 많이 사용되는 단어이다.

위의 예제에서는 이산변수인 바이너리 픽셀이기때문에 `sigmoid`를 통과시킬 수 있었지만, 임의의 연속변수를 모델링할 때는 `가우시안 혼합(Gaussian mixture) 모델`을 이용할 수 있다.

