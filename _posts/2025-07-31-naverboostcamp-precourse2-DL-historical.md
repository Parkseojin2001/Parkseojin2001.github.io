---
title: "[BoostCamp AI Tech / Pre-Course 2] 딥러닝 기본 - 딥러닝의 역사"
description: "네이버 부스트코스의 Pre-course 강의를 기반으로 작성한 포스트입니다."

categories: [NAVER BoostCamp AI Tech, Pre-Course]
tags: [Naver-Boostcourse, Pre-Course, pytorch]

permalink: /boostcamp/pre-course/DL-historical/

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-07-31
last_modified_at: 2025-07-31
---
## 인공지능 소개
---------

인공지능의 정의는 사람의 지능을 모방하는 것이라고 말할 수 있다.

<img src="https://cdn.inflearn.com/public/files/pages/b765bb2b-f098-419a-ad7b-38c4d8cd440d/%EA%B0%9C%EB%85%90.png" width="400" height="300">

### 딥러닝의 Key Component

- The data that the model can learn from
    - 다루고자 하는 데이터
    - 풀고자 하는 문제의 타입에 따라 달라진다.
        - 문제 타입 : Classification, Sementic Segmentation, Detection, Pose Estimation, Visual QnA 등
- The model how to transform the data
    - 데이터를 변형하여 원하는 결과를 도출하는 모델
    - AlexNet, GoogLeNet, ResNet, DenseNet, LSTM, Deep Auto Encoders, GAN 등
- The loss function that quantifies the badness of the model
    - 모델을 학습시키기 위한 손실함수
    - 상황에 따라 사용하는 손실함수가 다르다.
        - Regression Task : MSE(Mean Squared Error) - [오차제곱]의 평균
        - Classification Task : CE(Cross Entropy) - [정답레이블*로그 추정치]의 평균
        - Probabilistic Task : MLE(Maximum likelihood Estimation) 또는 MSE
    - 그러나 고정된 것은 아니며, 각 손실함수의 특징과 문제의 특징을 잘 고려하여 그때그때 선택해야한다.
- The algorithm to adjust the parameters to minimize the loss
    - 목적을 달성하기 위해 손실함수를 최소화시키는 알고리즘
    - SGD, Momentum, NAG, Adagrad, Adadelta, Rmsprop 등
    - 손실함수를 단순히 최소화시키는 것이 아니라, 학습하지 않은 데이터에서 잘 동작하도록 하는 것이 중요하다.
        - Dropout, Early stopping, k-fold validation, Weight decay, Batch normalization, MixUp, Ensemble, Bayesian Optimization 등 

위의 4가지가 딥러닝의 중요한 4요소라고 볼수 있으며, 새로운 논문을 이해할 때 저 4가지에서 어떤 차이점이 있는지를 중점적으로 보게 된다.

## 딥러닝의 역사
---------------

[Deep Learning's Most Important Ideas - A Brief Historical Review(2020-07-29, Denny Britz)](https://dennybritz.com/posts/deep-learning-ideas-that-stood-the-test-of-time/)를 참고하였습니다.

> 2012 - AlexNet
>
> 2012년 처음으로 Alex-net이라는 이미지넷 대회에서 딥러닝이 우승했다. 그 이후로는 한번도 딥러닝이 아닌 다른 기법이 우승한 적이 없다.

<img src="https://resources-public-blog.modulabs.co.kr/blog/prd/content/259481/Untitled-2.png">

그 이후로 이론적으로만 연구되던 딥러닝이 실제로 사용되기 시작했고, 패러다임의 변화가 일어났다.

> 2013 - DQN
>
> 추후에 알파고를 만들게 되는 딥마인드가, 아타리 게임을 클리어하기 위하여 [Q-Learning](https://ko.wikipedia.org/wiki/Q_%EB%9F%AC%EB%8B%9D)을 딥러닝에 접목한 DQN을 사용하고, 논문을 냈다. 딥마인드는 이 결과를 눈여겨본 구글에 인수되었다.

<img src="../assets/img/post/naver-boostcamp/DQN.png">

> 2014 - Encoder / Decoder, Adam Optimizer
>
> [NMT(Neural Machine Translation)](https://ko.wikipedia.org/wiki/%EC%8B%A0%EA%B2%BD%EB%A7%9D_%EA%B8%B0%EA%B3%84_%EB%B2%88%EC%97%AD)를 풀기위해 Encoder / Decoder가 등장했다.

<img src="../assets/img/post/naver-boostcamp/encoder-decoder.png">

특정 언어의 시퀀스를 Encoder가 어떤 벡터에 Encoding하고, Decoder가 이 벡터를 다른 언어 시퀀스로 Decoding 시켜준다.

Adam(Adaptive Momentum) Optimizer가 등장했다.

기존의 SGD, Adam 등의 Optimizer에 비하여 Optimizing 결과가 좋아서, 많이 사용된다.

- 여기에 이유가 설명되어 있지는 않다.
- 컴퓨팅 리소스가 충분한 Google 등의 테크기업들이 실험결과를 제공한 것을 바탕으로, '그냥 그렇게 하면 좋더라...'의 느낌으로 사용한다.

> 2015 - GAN
> 
> 이안 굿펠로우의 GAN(Generative Adversarial Network) 논문이 발표되었다.
> Generator와 Discriminator를 두고, 서로 경쟁시킴으로써 '그럴듯한' 결과를 만들어내는 기술이다.

> 2015 - ResNet
> 
> Deep Learning은 Shallow Network를 활용하지 않고, 신경망의 레이어를 많이 쌓는 방식이다(그렇다고 알려져 있다). 그러나 ResNet이 나오기 이전까지는, 레이어를 너무 깊게 쌓으면 테스트 데이터에 대한 성능이 좋지 않다고 알려져 있었다.

<img src="https://oopy.lazyrockets.com/api/v2/notion/image?src=https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Fd9e55bae-a991-429c-a39c-da6c986dcac4%2FUntitled.png&blockId=2e3785a8-b9b8-4237-9bef-7f85141f3dbc" width="400" height="600">

ResNet은 100단이 넘어가는 수의 레이어를 쌓아도 성능이 개선되는 것을 보여주면서, '딥'러닝으로의 진정한 패러다임 전환이 일어났다.

> 2017 - Transformer
>
> [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) 논문이 발표되었다. Transformer 구조 또는 Attention 구조는 굉장히 중요한 파트이고, 왜 기존의 방법들에 비해 높은 성능을 내는지 알아두는 것이 좋다.

<img src="https://miro.medium.com/v2/resize:fit:380/0*9FPmENBVFlyvAoGP.png">

발표 직후에는 domain specific한 방법이었지만, 기존의 CNN과 RNN 등 대부분의 방법을 대체하면서 현재는 Vision까지 넘보고 있다.


> 2018 - Bert
>
> Bidirection Encoder Representations from Transformers의 약자로, 2017년 발표된 Transformer 구조를 활용하되, Bidirectional Encoder를 사용한다.

<img src="https://velog.velcdn.com/images/tm011899/post/b70bc6f6-2b06-4ae7-b73d-96cbb4f11950/image.jpg">

Bert에서 주목해야 할 것은 Bert 그 자체보다 Fine Tuned NLP Model의 등장이다.

Language 모델은 이전에 단어가 주어졌을 때 다음에 어떤 단어가 나올지를 맞추는 것인데, 이를 이용해 그럴싸한 문장을 만들어 내기 위해서는 해당 도메인과 관련된 수많은 말뭉치가 필요했다. 그러나 Fine Tuned NLP Model은 먼저 세상에 널려있는 수많은 말들로 Pre-tuning을 하고, 그 이후 해당 도메인의 말뭉치로 Fine-tuning을 한다.

> 2019 - Big Language Models(GPT-X)
>
> OpenAI에서 발표한 GPT-3 모델은 Bert의 끝판왕이라고 볼 수 있다. 쉽사리 학습할 수 없는 굉장히 많은(175B) 파라미터를 가지고 있다.

<img src="https://techrecipe.co.kr/wp-content/uploads/2020/06/200601_openai_001.png">

> 2020 - Self Supervised Learning
>
> Self Supervised Learning은 Label을 모르는 unsupervised data를 활용할 수 있다.
>
> [SimCLR : a simple framework for contrastive learning of visual representations](https://arxiv.org/pdf/2002.05709)

한정된 학습데이터가 주어졌을 때, 여러 번 수정을 해가며 가장 결과가 좋은 모델을 수동적으로 찾아내는것이 일반적인 방법이었다.

<img src="https://rauleun.github.io/assets/images/210410-SimCLR/1-SimCLR.png">

SimCLR 논문에서는 어떻게 좋은 Visual Representation(이미지를 벡터로 바꾸는것)을 할 수 있을 것인가를 다룬다.

최근에는 이 논문을 기반으로 [BYOL(Bootstrap Your Own Latent)](https://arxiv.org/pdf/2006.07733)와 같은 핫한 논문들이 나오고 있다.

또, 고도화된 도메인 지식이 있을 때 학습데이터를 시뮬레이터로 추가로 만들어내는, self supervised data sampling과 같은 기법도 등장하고 있다.