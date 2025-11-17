---
title: "Image Generation"
description: "생성형 이미지 모델에 대한 종류와 발전과 GANs, AEs, 그리고 Diffusion Models에 대하여 대표적인 방법에 대한 내용을 정리한 포스트입니다."

categories: [Naver-Boostcamp, Generative AI]
tags: [Generative AI, LLM, Image Generation]

permalink: /naver-boostcamp/generative-ai/05

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-11-05
last_modified_at: 2025-11-05
---

## Generative Adversarial Networks, GANs
-----

`GANs`으ㄴ 판별자와 생성자를 적대적으로 학습하는 모델 구조이다.

- 판별자: 입력 이미지가 생성된 이미지인지 진짜 이미지인지 판별
    - 생성 여부를 잘 판단하도록 학습 &rarr; maximization
- 생성자: 잠재 변수 $z$ 를 입력으로 받아 학습 데이터의 분포에 가까운 이미지를 생성
    - 판별자가 생성 여부를 판단하지 못하도록 학습 &rarr; minimization

$$
L_{\text{GAN}} = \underset{G}\ {\text{min}}\underset{D}{\text{max}} V(D,G) = \mathbb{E}_{x \sim p_{data(x)}}[log \ D(x)] + \mathbb{E}_{x \sim p_{z}(z)}[log \ (1 - D(G(x)))]
$$

GANs는 모델이 수렴하기 전에는 데이터의 분포가 다르지만 모델 수렴 후에는 데이터의 분포가 같아지는 과정을 거친다.

### cGAN, Pix2Pix

- Conditional Generative Adversarial Networks
    - GAN 학습에 조건을 주입하여 학습하는 구조
    - 주어진 조건에 따라 이미지 생성이 가능하도록 함

    $$
    L_{\text{GAN}} = \underset{G}\ {\text{min}}\underset{D}{\text{max}} V(D,G) = \mathbb{E}_{x \sim p_{data(x)}}[log \ D(x | y)] + \mathbb{E}_{x \sim p_{z}(z)}[log \ (1 - D(G(x|y)))]
    $$

- Pix2Pix
    - 이미지를 조건으로 이미지를 변환하는 방법
    - 학습을 위해 서로 매칭되는 paired image가 필요함
    - Conditional GAN 구조를 따르며 조건으로 이미지가 변영됨
        - ex. edge 이미지를 조건으로 생성자 G를 통해 사진을 생성하고 판별자 D는 생성된 사진 또는 진짜 사진을 판별하도록 학습



### CycleGAN, StarGAN

- CycleGAN
    - Pix2Pix 방식은 paired images가 필요하지만 많은 이미지를 확보하기 어려움
    - CycleGAN에서는 unpaired images로 학습하기 위해 cycle consistency loss를 제안함
    - Unpaired images로 학습하여 이미지를 원하는 형태로 변환하는 것이 가능한 방법
    - Unpaired image $(X, y)$ 를 학습하기 위해 두 개의 생성자 $G, F$ 와 판별자 $D_{x}, D_{y}$ 를 학습에 사용

- StarGAN
    - 여러 도메인을 생성 모델에 반영하기 위해 많은 도메인 별 생성 모델이 필요하고 학습 효율성이 떨어짐
    - 이를 개선하기 위해 단일 생성 모델만으로 여러 도메인을 반영할 수 있는 구조를 제안함
    - CycleGAN에서 제안된 cycle consistency loss와 domain classification을 활용하여 여러 도메인을 반영할 수 있는 모델 구조
    - 목적 함수
        - $L_{\text{GAN}}$ : adversarial training loss
        - $L_{\text{cls}}$ : 도메인을 판단하기 위한 loss
        - $L_{\text{rec}}$ : cycle consistency loss
    - 학습에 반영된 여러 도메인에 따라 이미지 생성이 가능함


### ProgressiveGAN, StyleGAN


- ProgressiveGAN
    - 고해상도 이미지 생성 모델을 학습하기 위해서는 많은 비용이 발생함
    - 고해상도 이미지를 생성하기 위해 저해상도 이미지 생성 구조부터 단계적으로 증강하는 모델 구조를 제안하여 적은 비용으로 빠른 수렴이 가능한 모델 구조를 제안
- StyleGAN
    - ProgressiveGAN 구조에서 style을 주입하는 방법을 제안
    - 잠재 공간 𝒵𝒵를 바로 사용하는 것이 아닌 mapping network $f$ 를 사용하여 변환된 𝒲𝒲를 입력으로 사용

## Autoencoders, AEs
------

## Diffusion Models
-------
