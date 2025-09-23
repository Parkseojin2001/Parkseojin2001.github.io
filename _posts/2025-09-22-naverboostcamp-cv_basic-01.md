---
title: "CNN to ViT"
description: "CNN 모델과 ViT 모델의 등장과 구조에 대한 내용에서 더 나아가 ViT를 이용한 self-supervised 학습 방법을 정리한 포스트입니다."

categories: [Deep Learning, CV]
tags: [CV, CNN, ResNet, VGG, Transformer, ViT]

permalink: /naver-boostcamp/computer-vision/01

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-09-22
last_modified_at: 2025-09-22
---

## Image Classification
-------

분류기(Classifier)는 입력을 어떤 카테고리 값과 매핑시켜 내보내는 장치이다. 이미지 분류는 이 분류기의 입력값으로 시각적 데이터만을 사용하여 추론하는 것을 일컫는다.

극단적으로 생각해보았을 때, 모든 분류 문제는 세상의 모든 시각적 데이터를 가지고 있다면 아주 쉽게 해결된다. 그냥 모든 데이터들 사이에서 비슷한 것들끼리 모으기만 하면 된다.

<img src="https://blogik.netlify.app/static/5680ba7394c320efaa16adfe9b5a1cf2/0a47e/k-nn.png">

즉, K Nearest Neighbors(K-NN) 문제로 해결할 수 있다. K-NN 문제는 단순히 이미지 레이블 데이터 값을 주위의 다른 데이터 레이블들과 비교하여 가장 비슷하다고 판단되는 후보군으로 편입시키는 문제이다. 이렇게 해결하는 분류기가 있다면, 마치 검색엔진처럼 작동한다. 

그러나, 이러한 접근 방식은 불가능하다. Time/Memory Complexity 무한대일 것이라는 점과, '비슷하다'는 기준을 어떻게 잡을건지가 모호하다는 것이 결정적인 불가능 요인이다. 따라서 컴퓨터 비젼은 방대한 데이터를 제한된 complexity의 시스템(인공 신경망)이라는 분류기에 녹여넣는 것이 목표이다.

### Fully Connected Layer Network

이런 이미지 분류를 가장 간단한 형태의 인공 신경망 분류기, 즉 단일 계층의 Fully Connected Layer Network로 구현했다고 생각해보자.

<img src="https://blogik.netlify.app/static/9c28f194fefe28a0fad68d5d88ec2be5/2bef9/fully-connected-layer-network.png">

이미지를 모델이 인식하는데 있어 single fully connected layer 방식을 사용하면 다음과 같은 문제가 발생한다.

<img src="https://blogik.netlify.app/static/69fce1d11ae55b460b98d0c216c66f99/2bef9/cropped-fully-connected-layer-network.png" width="500" height="500">

- 같은 클래스에 해당하는 이미지이지만 이전에 학습된 이미지와 달라지면 이를 제대로 인식하지 못한다.
    - ex. 학습은 말의 앞모습 전체을 가지고 학습을 했지만, 말의 머리 모양만 crop해서 모델 테스트로 입력하면 말이라고 인식하지 못한다.
- 각 픽셀마다 대응하는 파라미터가 있어야 하므로 굉장히 많은 파라미터가 필요하여 모델의 크기가 굉장히 크다.

## Convolutional Neural Network(CNN)
---------------

이러한 Fully Connect Layer Network를 해결하기 위해 고안해낸 아이디어가 `CNN(Convolutional Neural Networks)` 이다.

<img src="https://blogik.netlify.app/static/ec56970b36992c0cc8866dfdaa8965d7/f6386/fully-vs-locally.png">

CNN은 모든 노드들을 다음 계층으로 전연결시키는 것이 아니라, `국소적인 연결(locally connect)`을 사용한다. 동일한 국소적 sliding window를 이미지의 모든 부분에 대입시켜 feature들을 뽑아냄으로써, 치우쳐 있는 이미지나 잘린 이미지라도 feature를 추출할 수 있고, 파라미터를 재활용하여 메모리도 적게 사용할 수 있다. 

이런 장점 때문에 많은 CV task의 backbone으로 활용되고 있다.

### CNN 아키텍처의 종류

아래의 그림은 CNN 모델의 발전 과정이다.

<img src="https://miro.medium.com/v2/resize:fit:640/format:webp/0*1SDlsJ7snNv_deec.png">

- `LeNet-5`
    - 1998년 Yann LeCun이 발표
    - 구조: Conv - Pool - Conv - Pool - FC - FC
    - Convolution size: $5 \times 5$ 필터 + 1 stride
    - Pooling: 2 $\times$ 2 max pooling + 2 stride

    <img src="https://blog.kakaocdn.net/dn/bChW5M/btqTKLpO6ST/bfZ99UcRuKz1xC7LwgrtB0/img.png">

- `AlexNet`
    - LeNet에서 모티베이션을 따왔다.
    - 파라미터와 학습 데이터를 훨씬 더 크게 늘렸다(605k neurons, 60 million paramters).
    - 필터 사이즈가 $11\times11$ 로 아주 크다. 최근에는 이런 큰 필터를 사용하지 않는다.
    - 활성화 함수를 `ReLU`를 사용하고, `dropout` 정규화 기법을 사용했다.
    - 논문에는 메모리 문제로 두 GPU에 올려서 학습했으며, 그 당시 명암을 조정하기 위해 사용했던 LRN(Local Response Normalization) 기법은 현재는 사용하지 않는다.
    - 구조 : Conv - Pool - LRN - Conv - Pool - LRN - Conv - Conv - Conv - Pool - FC - FC - FC

    <img src="https://resources-public-blog.modulabs.co.kr/blog/prd/content/259481/Untitled-2.png">

- `VGGNet`
    - 3x3의 작은 필터와 2x2 max pooling 사용, LRN 제거로 아키텍쳐가 비교적 간단해졌으나 성능은 더 좋아졌다.
    - 19 layer로 AlexNet(12 layer)보다 더 깊다.
        - 작은 필터크기임에도 불구하고, 더 깊이 층을 쌓아 receptive field의 크기를 키웠다.
    - 미리 학습된 feature를 fine-tuning하지 않고도 다른 task에 적용 가능할 정도로 일반화가 잘 되었다.

> **Receptive field in CNN**<br>
> Receptive Field(수용 영역)은 컨볼루션 신경망(CNN)에서 출력 레이어의 뉴런 하나에 영향을 미치는 입력 뉴런들의 공간 크기를 의미한다.
> 만약, $K \times K$ conv과 stride 1 그리고 pooling layer size가 $P \times P$ 라면 수용 영역은 $(P + K - 1) \times (P + K - 1)$ 다.
>
> <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdna%2FK2LSj%2FbtsEkwNGPSb%2FAAAAAAAAAAAAAAAAAAAAAD7nVGWaWUAzWCByJoAGuZ2vtPJHfiGxZCitpCHjXDC_%2Fimg.png%3Fcredential%3DyqXZFxpELC7KVnFOS48ylbz2pIh7yKj8%26expires%3D1759244399%26allow_ip%3D%26allow_referer%3D%26signature%3DI5Cocax1iAcInfxmSMwxUZ%252FCNQY%253D" width="500" height="300">
>
> Receptive field는 CNN 이미지를 인식하는 방식을 이해하는데 중요한 역할을 하는데 이 때 필드가 클수록 더 복잡한 특징을 추출할 수 있지만 출력 이미지의 해상도는 낮아진다.

AlexNet에서 VGGNet으로 발전하면서, 더 깊은 네트워크일수록 더 좋은 성능을 낸다는 것을 확인했다. 그렇다면 과연 층을 단순히 더 깊게 쌓으면, 항상 더 좋은 네트워크를 얻을 수 있을까? 물론 그렇지 않았다. 층을 깊게 쌓으면 쌓을수록 학습을 어렵게 만드는 문제들이 있었다.

- **기울기 소실/폭발(Gradient vanishing/exploding)**
- 연산 복잡도 증가(Computationally complex)
- depth가 어느정도 깊어지면, 성능이 떨어지기 시작
    - ~~Overfitting~~ 이 아닐까? &rarr; 기울기 소실/폭발로 인해 학습이 더 진행되지 않음

이러한 문제점들을 인식한 채로 새로운 네트워크 형태들이 등장하기 시작했다.

- `GoogLeNet`
    - ***Inception Module*** 을 여러 층 쌓는 형태를 제안한다.
        - 하나의 층에서 다양한 크기의 필터를 사용하여 여러 측면에서에서 살펴본다(depth 확장이 아닌 수평 확장)
        - 이 결과들은 모두 concatenation 하여 다음 층으로 넘겨주게 된다.
        - 이 때, $1 \times 1$ conv를 한 번 적용해 채널 수를 줄여 계산 복잡도를 떨어뜨린다. 이를 병목(bottleneck) 층이라고 한다.
            - $3 \times 3$, $5\times 5$ conv 직전 / pooling 연산 직후
- `ResNet`
    - 최초로 100개가 넘는 layer를 쌓음으로써, 더 깊은 layer를 쌓을수록 성능이 더 좋아진다는 것을 보여준 첫 모델이다. 또한, 인간의 지각 능력을 뛰어넘은 첫 모델이기도 하다.
    - 계기
        - 네트워크 깊이를 늘리다보면 어느 순간부터 정확도(accuracy) 감수가 포화 상태(saturated)에 이른다.
        - 기존 인식(가설)
            - 모델 파라미터가 너무 많아지면 overfitting 되어 training error가 더 적고 test error가 더 많은 결과가 나올 것이다.

        <img src="https://media.geeksforgeeks.org/wp-content/uploads/20200424200128/abc.jpg">

        - 실험 결과
            - **overfitting 문제가 아니라**,  training error든 test error든 더 깊은 층(56)의 네트워크가 더 얕은 층(20)의 네트워크보다 에러 수가 높게 나온다.
            - **degradation 문제이다. 기울기 소실 때문에 최적화(optimization)이 덜 되어 깊은 층의 네트워크가 학습이 덜 된 것이다!**
    
    - ***residual(skip) connection*** : 층이 깊어질수록 기존의 input x의 영향력(기울기)이 소실되어 충분히 학습하기 어렵다. 따라서, 몇 개의 층을 지나면 기존의 x와 동일한 값(identity)를 잔차(residual)에 더해주어, 잔여부분만 학습함으로써 학습 부담을 경감시킨다.

        <img src="https://wikidocs.net/images/page/164800/Fig_03.png">

        - 역전파 시에도 gradient가 원래 네트워크 레이어 쪽과 skip connection 쪽 두 군데로 흐르므로, 한 곳에서 기울기 소실이 일어나더라도 다른 한쪽을 통해 학습을 정상적으로 지속할 수 있게 된다.
        - skip connection이 한번 일어날 때마다 역전파 gradient가 흐르는 방법의 경우의 수가 2배로 늘어나므로, 전체 경우의 수는 $2^n$ 개가 된다.
        - residual block은 2개의 3x3 conv layer로 이루어져 있다.
    - 출력 직전 FC 층은 하나만 존재한다.

- `EfficientNet`
    - 기존에 네트워크 성능을 높이는 방법
        - deep / wide / (high) resolution scaling
            - high resolution scaling : 애초에 input 이미지의 resolution이 높으면 성능이 더 좋아진다.
        - 그러나 세 방법은 각각 accuracy saturate의 시점이 다르다.
    - 그래서 세 방법 모두를 적절히 섞은 compound scaling이 등장한다.
    - 지금까지 나왔던 모든 방식들을 대상으로, 적은 FLOP에서도 압도적인 성능차를 보였다.

    <img src="https://blog.kakaocdn.net/dna/m1RiB/btq2gP3Zst5/AAAAAAAAAAAAAAAAAAAAAEOanue_GJt4eZFDsXhcLrmchjzuwCIDRF7QrJccjD5x/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1759244399&allow_ip=&allow_referer=&signature=FmZczH8byDFGCq7vnLgL33LcR6w%3D">

참고: [Modern Convolutional Neural Networks](https://parkseojin2001.github.io/boostcamp/pre-course/cnn-2/)

## Vision Transformers
---------

