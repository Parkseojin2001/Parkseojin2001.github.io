---
title: "PyTorch 활용하기"
description: "네이버 부스트코스의 Pre-course 강의를 기반으로 작성한 포스트입니다."

categories: [Naver-Boostcamp, Pre-Course 2]
tags: [Naver-Boostcamp, Pre-Course, pytorch]

permalink: /boostcamp/pre-course/pytorch-advance/

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-07-26
last_modified_at: 2025-07-26
---

## Multi-GPU 학습
----------

오늘날의 딥러닝은 엄청난 데이터를 다뤄야 하며 이를 위해 많은 GPU를 사용한다. 




## Hyperparameter Tuning
------------

모델 스스로 학습하지 않는 값은 사람이 지정하며 이를 하이퍼파라미터라고 한다. 
- learning rate
- 모델 크기
- optimizer

하이퍼 파라미터에 의해서 값이 크게 좌우될 때도 있지만 요즘은 크게 변화하지는 않는다.

그러므로, 마지막 0.01의 성능을 올릴 때 도전해 볼만한 방법이다.

가장 기본적은 방법은 grid와 random이 있으며 최근에는 베이지안 기반 기법들이 주도한다.

<img src="../assets/img/post/naver-boostcamp/grid_random_layout.png">

### Ray

- multi-node multi processing 지원 모듈
- ML/DL의 병렬 처리를 위해 개발된 모듈로 현재 분산 병렬 ML/DL 모듈의 표준이다.
- Hyperparameter Search를 위한 다양한 모듈을 제공한다.

```python
data_dir = os.path.abspath("./data")
load_data(data_dir)
config = {
    "l1": tune.sample_from(lambda_: 2**np.random.randint(2, 9)),
    "l2": tune.sample_from(lambda_: 2**np.random.randint(2, 9)),
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([2, 4, 8, 16])
}

scheduler = ASHAScheduler(
    metric="loss", mode="min", max_t=max_num_epochs, grace_period=1, reduction_factor=2
)
reporter = CLIReporter(
    metric_columns=["loss", "accuracy", "training_iteration"]
)
result = tune.run(
    partial(train_cifar, data_dir=data_dir),
    resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
    config=config,
    num_samples=num_samples,
    scheduler=scheduler,
    progress_reporter=reporter
)
```
