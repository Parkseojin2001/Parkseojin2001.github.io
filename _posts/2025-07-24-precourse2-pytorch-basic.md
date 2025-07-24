---
title: "PyTorch 구조 학습하기"
description: "네이버 부스트코스의 Pre-course 강의를 기반으로 작성한 포스트입니다."

categories: [Naver-Boostcamp, Pre-Course 2]
tags: [Naver-Boostcamp, Pre-Course, pytorch]

permalink: /boostcamp/pre-course/pytorch-basic/

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-07-24
last_modified_at: 2025-07-24
---

## AutoGrad & Optimizer
-----------

### torch.nn.Module

딥러닝을 구성하는 Layer의 base class로 Input, Output, Forward, Backward를 정의한다.

학습의 대상인 parameter(tensor)도 정의한다.

<img src="https://camo.githubusercontent.com/62582fbb579217fb2c7187d9a4fc54b479032268c2d0916c192880655404edfa/68747470733a2f2f63646e2d696d616765732d312e6d656469756d2e636f6d2f6d61782f313630302f312a71314d374c47694454697277552d344c634671375f512e706e67">

### nn.Parameter

Tensor 객체의 상속 객체로 nn.Module 내에 attribute가 될 때는 `required_grad=True`로 지정되어 학습 대상이 되는 Tensor이다.

대부분의 layer에 weights 값들이 지정되어 있기 때문에 직접 지정할 일은 잘 없다.

```python
class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(
            torch.randn(in_features, out_features)
        )
        self.bias = nn.Parameter(torch.randn(out_features))
    
    def forward(self, x: Tensor):
        return x @ self.weights + self.bias

```

### Backward

Layer에 있는 parameter들의 미분을 수행한다.
Forward의 결과값(model의 output=pred)과 실제값 간의 차이(Loss)을 이용하여 Loss값이 작아지는 Parameter로 업데이트하는 과정을 거친다.

```python
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    print(loss)
    loss.backward()
    optimizer.step()
```

실제 backward는 Module 단계에서 직접 지정이 가능하다.


## PyTorch Dataset & DataLoader
----------

모델에 데이터를 입력하는 과정은 아래와 같이 표현할 수 있다.

<img src="../assets/img/post/naver-boostcamp/pytorch_dataset.png">

### Dataset 클래스

데이터 입력 형태를 정의하는 클래스로 데이터를 입력하는 방식의 표준화한다.

Image, Text, Audio 등에 따른 다른 형식의 입력을 정의한다.

```python
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, text, labels):
        self.labels = labels    # 초기 데이터 생성 방법을 지정
        self.data = text
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels[idx]    # index 값을 주었을 때 반환되는 데이터의 형태 (X, y)
        text = self.data[idx]
        sample = {"Text": text, "Class": label}
        return sample
```

Dataset 클래스를 생성할 때 주의해야할 점이 있다.

- 데이터 형태에 따라 각 함수를 다르게 정의
- 모든 것을 데이터 생성 시점에 처리할 필요는 없음
    - image의 Tensor 변화는 학습에 필요한 시점에 반환
- 데이터 셋에 대한 표준화된 처리방법 제공 필요
    - 후속 연구자 또는 동료에게 도움이 됨
- 최근에는 HuggingFace 등 표준화된 라이브러리 사용

### DataLoader 클래스

DataLoader는 Data의 Batch를 생성해주는 클래스로 학습 직전(GPU feed 전) 데이터의 변환하는 일을 한다.

Tensor로 변환하고 Batch 처리가 메인 업무로 병렬적인 데이터 전처리 코드의 고민이 필요하다.

```python
text = ['Happy', 'Amazing', 'Sad', 'Unhappy', 'Glum']
labels = ['Positive', 'Positive', 'Negative', 'Negative', 'Negative']
MyDataset = CustomDataset(text, labels)     # Dataset 생성

MyDataLoader = DataLoader(MyDataset, batch_size=2, shuffle=True)    # DataLoader Generator
next(iter(MyDataLoader))
# {'Text': ['Glum','Sad'], 'Class': ['Nevative', 'Negative']}

for dataset in DataLoader:
    print(dataset)
# {'Text': ['Glum','Unhappy'], 'Class': ['Nevative', 'Negative']}
# {'Text': ['Sad','Amazing'], 'Class': ['Nevative', 'Positive']}
# {'Text': ['happy'], 'Class': ['Positive']}
```

```python
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)
```

## 모델 불러오기
----------

## Monitoring tools for PyTorch
-------------