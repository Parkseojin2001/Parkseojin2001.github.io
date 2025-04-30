---
title: "파이토치 기초(2)"
excerpt: "데이터 세트와 데이터 로더 / 모델 & 데이터세트 분리 / 모델 저장 및 불러오기"

categories:
  - Pytorch
tags:
  - [Pytorch]

permalink: /pytorch/basic-2/

toc: true
toc_sticky: true

date: 2025-04-30
last_modified_at: 2025-04-30
---

## 🦥 데이터세트와 데이터로더

데이터세트: 데이터의 집합을 의미하며, 입력값(X)과 결과값(Y)에 대한 정보를 제공하거나 일련의 데이터 묶음을 제공
- 구조: 일반적으로 데이터베이스(Database)의 테이블(Table)과 같은 형태
- 데이터세트의 한 패턴을 테이블의 행(Row)으로 간주한다면, 이 행에서 데이터를 불러와 학습을 진행

데이터의 구조나 패턴은 매우 다양하며 학습해야 하는 데이터가 파일 경로로 제공되거나 데이터를 활용하기 위해서 전처리 단계가 필요한 경우가 있다. 또한 다양한 데이터가 포함된 데이터세트에서는 특정한 필드의 값을 사용하거나 사용하지 않을 수 있다.

데이터를 변형하고 매핑하는 코드를 학습 과정에 직접 반영하면 **모듈화(Modularization)**, **재사용성(Reusable)**, **가독성(Readability)** 등을 떨어뜨리는 주요 원인이 된다. 이러한 현상을 방지하고 코드를 구조적으로 설계할 수 있도록 데이터세트와 데이터로더를 사용한다.

### 데이테세트

**데이터세트(Dataset)**는 학습에 필요한 데이터 샘플을 정제하고 정답을 저장하는 기능을 제공한다.
- 초기화 메서드(`__init__`): 입력된 데이터의 전처리 과정을 수행하는 메서드
  - 새로운 인스턴스가 생성될 때 학습에 사용될 데이터를 선언하고, 학습에 필요한 형태로 변형하는 과정을 진행
- 호출 메서드(`__getitem__`): 학습을 진행할 때 사용되는 하나의 행을 불러오는 과정
  - 입력된 색인(index)에 해당하는 데이터 샘플을 불러오고 반환
  - 초기화 메서드에서 변형되거나 개선된 데이터를 가져오며, 데이터 샘플과 정답을 반환
- 길이 반환 메서드(`__len__`): 학습에 사용된 전체 데이터세트의 개수를 반환
  - 메서드를 통해 몇 개의 데이터로 학습이 진행되는지 확인할 수 있음

```python
# 데이터세트 클래스 기본형
class Dataset:

  def __init__(self, data, *arg, **kwargs):
    self.data = data
  
  def __getitem__(self, index):
    return tuple(data[index] for data in data.tensors)
  
  def __len__(self):
    return self.data[0].size(0)

```

모델 학습을 위해 임의의 데이터세트를 구성할 때 파이토치에서 지원하는 데이터세트 클래스를 상속받아 사용한다. 새로 정의한 데이터세트 클래스는 현재 시스템에 적합한 구조로 데이터를 전처리해 사용한다.

### 데이터로더

**데이터로더(DataLoader)**는 데이터세트에 저장된 데이터를 어떠한 방식으로 불러와 활용할지 정의한다. 학습을 조금 더 원활하게 진행할 수 있도록 여러 기능을 제공한다.
- 배치 크기(batch_size): 학습에 사용되는 데이터의 개수가 매우 많아 한 번의 에폭에서 모든 데이터를 메모리에 올릴 수 없을 때 데이터를 나누는 역할을 한다.
  - 전체 데이터세트에서배치 크기만큼 데이터 샘플을 나누고, 모든 배치를 대상으로 학습을 완료하면 한 번의 에폭이 완료
- 데이터 순서 변경(shuffle): 데이터의 순서로 학습되는 것을 방지
- 데이터 로드 프로세스 수(num_workers): 데이터를 불러올 때 사용할 프로세스의 개수

### 다중 선형회귀

데이터세트와 데이터로더를 활용해 다중 선형회귀를 구현하면 아래와 같이 구현할 수 있다.

```python
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

# 기본 데이터 구조 선언
train_x = torch.FloatTensor([
  [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]
])
train_y = torch.FloatTensor([
  [0.1, 1.5], [1, 2.8], [1.9, 4.1], [2.8, 5.4], [3.7, 6.7], [4.6, 8]
])

# 데이터세트와 데이터로더
train_dataset = TensorDataset(train_x, train_y)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)

# 모델, 오차 함수, 최적화 함수 선언
model = nn.Linear(2, 2, bias=True)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 데이터로더 적용
for epoch in range(20000):
  cost = 0.0
  for batch in train_dataloader:
    x, y = batch
    output = model(x)

    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    cost += loss

  cost = cost / len(train_dataloader)

  if (epoch + 1) % 1000 == 0:
    print(f"Epoch: {epoch+1:4d}, Model: {list(model.parameters())}, Cost: {cost:.3f}")
```

실제 환경에서 적용되는 데이터(학습에 사용하지 않은 데이터)를 통해 지속적으로 검증하고, 최적의 매개변수를 찾는 방법으로 모델을 구성해야 한다. 이 이유로 데이터의 구조나 형태는 지속해서 변경될 수 있으므로 데이터세트와 데이터로더를 활용해 코드 품질을 높이고 반복 및 변경되는 작업에 대해 더 효율적으로 대처해야한다.

## 🦥 모델/데이터세트 분리