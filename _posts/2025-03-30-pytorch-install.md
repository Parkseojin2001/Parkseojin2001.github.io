---
title: "파이토치 설치"
description: "파이토치 소개 / 파이토치 설치"

categories: [Deep Learning, 파이토치 트랜스포머를 활용한 자연어 처리와 컴퓨터 비전 심층학습]
tags: [pytorch]

permalink: /pytorch-book/pytorch/installation/

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-03-30
last_modified_at: 2025-03-30
---

## 파이토치란?
---------

**파이토치(PyTorch)**는 딥러닝 및 인공지능 애플리케이션에 널리 사용되는 파이썬용 오픈 소스 머신러닝 라이브러리다.

## 파이토치 특징
---------

파이토치의 주요 기능은 다음과 같다.

- 동적 계산 그래프
- GPU 가속
- 사용하기 쉬움
- 우수한 성능
- 활발한 커뮤니티

몇 가지 제한사항과 잠재적인 단점이 있다.
- 제한된 프로덕션 지원
- 제한된 문서
- 호환성
- 제한된 통합

## 파이토치 설치
---------

파이토치는 다양한 모듈을 제공하며, 대표적으로 **자동 미분 시스템에 구축된 심층 신경망 라이브러리(pytorch), 영상 처리를 위한 이미지 변환 라이브러리(torchvision), 오디오 및 신호 처리를 위한 라이브러리(torchaudio)**가 있다.

### 파이토치 CPU 설치

일반적으로 파이토치는 CPU 환경에서 사용하지 않지만 주로 간단한 실험을 진행하거나 GPU 가속을 적용하기 어려운 환경에서 사용된다.

**패키지 매니저를 이용한 설치**<br>

```python
# 파이토치 CPU 설치(윈도우, 맥, 리눅스)
pip install torch torchvision torchaudio
```

**아나콘다를 이용한 설치**<br>

```python
# 파이토치 CPU 설치(윈도우, 리눅스)
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# 파이토치 CPU 설치(aor)
conda install pytorch torchvision torchaudio -c pytorch
```

파이토치의 설치된 버전을 확인할 수 있다.

```python 
# 파이토치 버전 확인
import torch

print(torch.__version__)
```

### 파이토치 GPU 설치

**윈도우 / 리눅스**<br>

GPU를 사용하기 위한 CUDA Toolkit와 NVIDIA CUDA 심층 신경망 라이브러리(cuDNN)을 설치한 후 pytorch를 설치한다.

```python
# 패키지 매니저를 이용한 설치
pip install torch torchvision torchaudio --index-url
https://download.pytorch.org/whl/cu118
```

```python
# 아나콘다를 이용한 설치
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

파이토치의 설치된 버전과 GPU 가속을 확인한다.

```python
# 파이토치 GPU 가속 확인
import torch

print(torch.__version__)
print(torch.cuda.is_available())
```

**맥**<br>

애플 실리콘에서는 **MPS(Metal Performance Shaders)**를 통한 GPU 가속을 적용한다. 맥에서는 파이토치 CPU를 설치하는 방법과 동일하다.


