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

## CNN architectures
-------

이미지를 모델이 인식하는데 있어 single fully connected layer 방식을 사용하면 다음과 같은 문제가 발생한다.

- 같은 클래스에 해당하는 이미지이지만 이전에 학습된 이미지와 달라지면 이를 제대로 인식하지 못한다.
    - ex. 학습은 말의 앞모습 전체을 가지고 학습을 했지만, 말의 머리 모양만 crop해서 모델 테스트로 입력하면 말이라고 인식하지 못한다.
- 각 픽셀마다 대응하는 파라미터가 있어야 하므로 굉장히 많은 파라미터가 필요하여 모델의 크기가 굉장히 크다.

이러한 문제점을 해결하기 위해 고안해내 아이디어가 `CNN(Convolutional Neural Networks)` 이다.

이 방식은 **지역적 특징을 학습하며 효율적인 파라미터 관리**가 가능하다.

이러한 방식은 많은 CV 태스크의 백본으로써 사용된다.

아래의 그림은 CNN 모델의 발전 과정이다.

<img src="https://miro.medium.com/v2/resize:fit:640/format:webp/0*1SDlsJ7snNv_deec.png">

### LeNet-5

LeNet-5는 1998년 Yann LeCun이 발표한 CNN 구조이다.

- 구조: Conv - Pool - Conv - Pool - FC - FC
- Convolution size: $5 \times 5$ 필터 + 1 stride
- Pooling: 2 $\times$ 2 max pooling + 2 stride

<img src="https://blog.kakaocdn.net/dn/bChW5M/btqTKLpO6ST/bfZ99UcRuKz1xC7LwgrtB0/img.png">

### AlexNet

`AlexNet`는 LeNet-5 이후에 나온 모델로 LeNet-5와 유사하지만, `AlexNet`은 7개의 layer로 더 큰 모델이다.

- 605k neurons
- 60 million paramters

<img src="https://resources-public-blog.modulabs.co.kr/blog/prd/content/259481/Untitled-2.png">

모델 구조를 2개로 나눈 이유는 GPU를 이용해 병렬로 학습하기 위함이다.

AlexNet 모델에서는 기존에 사용하지 않았던 `ReLU` 함수를 사용하고 과적합 방지를 위해 `dropout` 규제를 사용하였다.

> **Receptive field in CNN**<br>
> Receptive Field(수용 영역)은 컨볼루션 신경망(CNN)에서 출력 레이어의 뉴런 하나에 영향을 미치는 입력 뉴런들의 공간 크기를 의미한다.
> 만약, $K \times K$ conv과 stride 1 그리고 pooling layer size가 $P \times P$ 라면 수용 영역은 $(P + K - 1) \times (P + K - 1)$ 다.
>
> <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdna%2FK2LSj%2FbtsEkwNGPSb%2FAAAAAAAAAAAAAAAAAAAAAD7nVGWaWUAzWCByJoAGuZ2vtPJHfiGxZCitpCHjXDC_%2Fimg.png%3Fcredential%3DyqXZFxpELC7KVnFOS48ylbz2pIh7yKj8%26expires%3D1759244399%26allow_ip%3D%26allow_referer%3D%26signature%3DI5Cocax1iAcInfxmSMwxUZ%252FCNQY%253D" width="500" height="300">
>
> Receptive field는 CNN 이미지를 인식하는 방식을 이해하는데 중요한 역할을 하는데 이 때 필드가 클수록 더 복잡한 특징을 추출할 수 있지만 출력 이미지의 해상도는 낮아진다.

### VGGNet

`VGGNet`는 16 또는 19 레이어를 쌓은 CNN model이다.

### ResNet

### EfficientNet



## Vision Transformers
---------

