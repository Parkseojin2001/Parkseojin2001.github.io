---
title: "[Spark] Job·Stage·Task는 실제로 어떻게 나뉠까"
description: "「러닝 스파크」 2장에서 다룬 잡·스테이지·태스크가 실제로 어떻게 나뉘는지 코드를 실행한 결과로 정리한 포스트입니다."

categories: [Data Engineering, Spark]
tags: [spark]

permalink: /data-engineering/spark/execution-units/

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2026-08-13
last_modified_at: 2026-08-20
---

[2장 포스트](/data-engineering/learning-spark/02/)에서 "액션이 잡을 만들고, 잡은 스테이지로, 스테이지는 태스크로 나뉜다"고 정리했다. 이 글은 이 실행 단위가 실제로 어떻게 나뉘는지를 코드를 실행한 결과로 정리한다.

1. 코드는 어떤 단위로 쪼개져 실행될까?
2. 액션과 트랜스포메이션은 어떻게 구분할까?
3. 셔플이 일어나는지는 어떻게 판단할까?
4. 액션 1개 = 잡 1개일까?
5. `count()`는 결과만 세는 액션인데 왜 셔플이 생길까?
6. 셔플 뒤 스테이지의 태스크는 몇 개일까?
7. 태스크 수는 어떻게 확인할까?

## 실험 환경

---

2장과 같은 코드를 Spark 3.5.9, `local[4]`(코어 4개)에서 돌렸다. AQE(Adaptive Query Execution)는 기본값대로 켜 두고, 비교할 때만 껐다.

```python
from pyspark.sql import SparkSession, functions as F

spark = (SparkSession.builder.master("local[4]")
         .config("spark.sql.adaptive.enabled", "true")   # 비교할 때만 "false"
         .getOrCreate())

df = spark.range(0, 20_000_000)
narrow = df.filter(F.col("id") % 2 == 0).withColumn("bucket", F.col("id") % 10)
wide = narrow.groupBy("bucket").count()
```

`narrow`는 짝수만 남기므로 `bucket`은 0, 2, 4, 6, 8의 5가지 값만 가진다. 이 숫자는 6번 질문에서 다시 나온다. 글에 나오는 잡·스테이지·태스크 수는 이 코드를 실행해서 나온 값이다.

> **AQE(Adaptive Query Execution)란?**
>
> 실행 도중에 셔플 결과의 실제 크기를 보고, 남은 계획을 다시 세우는 기능이다. 스파크 3.2부터 기본으로 켜져 있다. 이 글에서는 AQE 때문에 생기는 두 가지 현상을 본다. 잡 수가 늘어나고(4·5번 질문), 셔플 뒤 태스크 수가 줄어든다(6번 질문).
{: .prompt-info }

## 1. 코드는 어떤 단위로 쪼개져 실행될까?

---

코드 → 잡 → 스테이지 → 태스크 → 코어 순서로 한 단계씩 확대하면 실행 단위가 어떻게 나뉘는지 보인다.

<div class="xd-legend">
<span><i class="xd-sw-job"></i>잡 (액션이 만듦)</span>
<span><i class="xd-sw-stage"></i>스테이지 (셔플로 나뉨)</span>
<span><i class="xd-sw-task"></i>태스크 (파티션 1개 = 태스크 1개)</span>
<span><i class="xd-sw-shuffle"></i>셔플 (파티션 간 데이터 교환)</span>
</div>

### 액션마다 잡이 생기고, 잡은 코드 순서대로 하나씩 실행된다

<div class="xd">
<div class="xd-timeline">
<div class="xd-code">df = spark.range(0, 20_000_000)</div><div class="xd-arrow">→</div><div class="xd-none">잡 없음 (기록만 함)</div>
<div class="xd-code">narrow = df.filter(...).withColumn(...)</div><div class="xd-arrow">→</div><div class="xd-none">잡 없음</div>
<div class="xd-code is-action">narrow.count()</div><div class="xd-arrow">→</div><div class="xd-jobs"><span class="xd-job">잡 0</span><span class="xd-job is-extra">잡 1 (AQE)</span></div>
<div class="xd-code">wide = narrow.groupBy("bucket").count()</div><div class="xd-arrow">→</div><div class="xd-none">잡 없음 (이 count()는 트랜스포메이션)</div>
<div class="xd-code is-action">narrow.limit(5).collect()</div><div class="xd-arrow">→</div><div class="xd-jobs"><span class="xd-job">잡 2</span></div>
<div class="xd-code is-action">wide.collect()</div><div class="xd-arrow">→</div><div class="xd-jobs"><span class="xd-job">잡 3</span><span class="xd-job is-extra">잡 4 (AQE)</span></div>
</div>
<div class="xd-caption">잡 번호는 실행했을 때 나온 번호다(0부터 시작). 점선 잡은 AQE가 셔플 앞뒤를 나눠 실행해서 추가로 생긴 잡이다(4·5번 질문).</div>
</div>

**잡끼리는 하나의 DAG로 묶이지 않는다.** 액션을 호출한 순서대로 잡이 하나씩 실행되고, **잡마다 자기 DAG를 따로** 만든다. 액션이 6개면 DAG도 6개 이상 생긴다.

### 잡 하나의 DAG 안에서 셔플 지점마다 스테이지가 나뉜다

위의 `wide.collect()`(잡 3·4)를 확대한 그림이다. 잡 0(`narrow.count()`)과는 별개다. 셔플이 1번 있으므로 스테이지는 2개다.

<div class="xd">
<div class="xd-dag">
<div class="xd-stage">
<div class="xd-stage-title">스테이지 1</div>
<div class="xd-ops"><span class="xd-op">range</span><span class="xd-op-arrow">→</span><span class="xd-op">filter</span><span class="xd-op-arrow">→</span><span class="xd-op">withColumn</span><span class="xd-op-arrow">→</span><span class="xd-op">bucket별로 미리 세기</span></div>
<small>좁은 연산은 데이터 교환이 없어서 한 스테이지로 묶임</small>
</div>
<div class="xd-shuffle"><div>셔플</div><div class="xd-bar"></div><div class="xd-mono">데이터 교환</div></div>
<div class="xd-stage">
<div class="xd-stage-title">스테이지 2</div>
<div class="xd-ops"><span class="xd-op">bucket별로 모음</span><span class="xd-op-arrow">→</span><span class="xd-op">count 합치기</span></div>
<small>다른 파티션의 데이터가 필요해서 셔플 뒤로 넘어감</small>
</div>
</div>
<div class="xd-chips">
<div class="xd-chip"><b>셔플 0번</b>스테이지 1개 (예: filter, select만)</div>
<div class="xd-chip"><b>셔플 1번</b>스테이지 2개 (예: groupBy 1번)</div>
<div class="xd-chip"><b>셔플 2번</b>스테이지 3개 (예: groupBy 후 orderBy)</div>
</div>
</div>

스테이지는 액션을 묶는 단위가 아니다. **잡 하나 안에서**, 그 잡의 연산들을 **셔플 경계로 자른 조각**이다. 그래서 스테이지 수는 액션 수가 아니라 잡 안에 셔플이 몇 번 있느냐로 정해진다.

### 스테이지는 파티션 수만큼의 태스크로 나뉜다

데이터가 파티션 6개로 나뉘어 있다면, 스테이지 1은 같은 코드를 파티션마다 실행하는 태스크 6개가 된다.

<div class="xd">
<div class="xd-split">
<div class="xd-part is-data">파티션 1</div><div class="xd-part is-data">파티션 2</div><div class="xd-part is-data">파티션 3</div><div class="xd-part is-data">파티션 4</div><div class="xd-part is-data">파티션 5</div><div class="xd-part is-data">파티션 6</div>
<div class="xd-down">↓ 파티션 하나당 태스크 하나</div>
<div class="xd-part is-task">태스크 1</div><div class="xd-part is-task">태스크 2</div><div class="xd-part is-task">태스크 3</div><div class="xd-part is-task">태스크 4</div><div class="xd-part is-task">태스크 5</div><div class="xd-part is-task">태스크 6</div>
</div>
</div>

**태스크 수는 코어 수가 아니라 파티션 수로 정해진다.** 코어가 몇 개든 이 스테이지의 태스크는 6개다.

### 코어 수는 태스크를 동시에 몇 개 돌리는지를 정한다

같은 태스크 6개를 코어 2개와 3개로 실행했을 때의 시간 흐름이다. 가로축이 시간이다.

<div class="xd">
<div class="xd-cores">
<div>
<div class="xd-core-title">코어 2개 <span>3번에 나눠 실행</span></div>
<div class="xd-lane"><span>코어 1</span><div class="xd-slot is-task">T1</div><div class="xd-slot is-task">T3</div><div class="xd-slot is-task">T5</div></div>
<div class="xd-lane"><span>코어 2</span><div class="xd-slot is-task">T2</div><div class="xd-slot is-task">T4</div><div class="xd-slot is-task">T6</div></div>
<div class="xd-axis"><span></span><span>1차</span><span>2차</span><span>3차</span></div>
</div>
<div>
<div class="xd-core-title">코어 3개 <span>2번에 나눠 실행</span></div>
<div class="xd-lane"><span>코어 1</span><div class="xd-slot is-task">T1</div><div class="xd-slot is-task">T4</div><div class="xd-slot is-empty">끝</div></div>
<div class="xd-lane"><span>코어 2</span><div class="xd-slot is-task">T2</div><div class="xd-slot is-task">T5</div><div class="xd-slot is-empty">끝</div></div>
<div class="xd-lane"><span>코어 3</span><div class="xd-slot is-task">T3</div><div class="xd-slot is-task">T6</div><div class="xd-slot is-empty">끝</div></div>
<div class="xd-axis"><span></span><span>1차</span><span>2차</span><span></span></div>
</div>
</div>
</div>

태스크가 일감이라면 코어는 일하는 사람이다. 사람이 늘어도 일감 수는 그대로이고, 끝나는 시간만 빨라진다. 코어가 4개라면 태스크 6개를 4개, 2개로 나눠 실행하니 두 번째 차례에 코어 2개가 논다. 그래서 파티션 수는 최소 코어 수 이상, 보통 코어 수의 2~3배로 맞춘다.

> **예외처럼 보이는 경우:** `spark.range()`처럼 파티션 수를 정하지 않으면 로컬 모드에서는 코어 수가 파티션 수가 된다. 그래서 `local[4]`에서는 태스크가 4개다. 이때도 규칙은 "태스크 수 = 파티션 수" 그대로다.
{: .prompt-tip }

## 2. 액션과 트랜스포메이션은 어떻게 구분할까?

---

먼저 **무엇을 돌려주는지** 보면 된다.

| 돌려주는 것 | 종류 | 예 |
|---|---|---|
| 새 **데이터 프레임** (계획) | 트랜스포메이션 | `filter()`, `select()`, `groupBy().count()` |
| 실제 **값**이나 출력 | 액션 | `df.count()` → 정수, `collect()` → 리스트, `show()` → 화면 출력 |

데이터 프레임을 돌려준다는 건 아직 계산할 필요가 없다는 뜻이다. 숫자나 리스트를 돌려주려면 지금 계산해야 한다. 변수에 저장하는지는 기준이 아니다.

```python
wide = narrow.groupBy("bucket").count()   # 트랜스포메이션 → 실행 안 됨
n    = narrow.count()                     # 액션 → 바로 실행됨, n = 10000000
```

둘 다 변수에 저장했지만, `wide`에는 계획이 들어가고 `n`에는 정수가 들어간다.

> **이름이 같아도 종류가 다를 수 있다.** `df.count()`는 정수를 돌려주는 **액션**이다. 하지만 `df.groupBy("bucket").count()`의 `count()`는 그룹별 개수 컬럼을 가진 데이터 프레임을 돌려주는 **트랜스포메이션**이다.
{: .prompt-warning }

## 3. 셔플이 일어나는지는 어떻게 판단할까?

---

기준은 하나다. **결과 파티션 하나를 만들 때 다른 파티션의 데이터가 필요한가?**

- 필요 없음 → 각 파티션이 자기 데이터만 보고 처리 → 셔플 없음 (좁은 의존성)
- 필요함 → 데이터를 파티션 사이로 옮겨야 함 → **셔플** (넓은 의존성)

`filter(id % 2 == 0)`은 행 하나만 보고 남길지 정한다. 반면 `groupBy("bucket")`은 같은 bucket 값이 여러 파티션에 흩어져 있으니 한곳에 모아야 한다.

| 셔플 없음 | 셔플 있음 |
|---|---|
| `select`, `filter`, `withColumn` | `groupBy(...).count()` 같은 그룹 집계 |
| `limit` | `orderBy`, `join`, `distinct` |
| | `df.count()` 같은 전체 집계 |

셔플은 `groupBy`처럼 **트랜스포메이션**이 만든다. 액션은 실행을 시작할 뿐이다. 예외는 `df.count()`처럼 **액션 자체가 집계**인 경우로, 5번 질문에서 본다.

## 4. 액션 1개 = 잡 1개일까?

---

아니다. 액션 하나가 잡을 **1개 이상** 만든다. 실행 결과는 이렇다.

| 액션 | 잡 (AQE 켜짐) | 잡 (AQE 꺼짐) |
|---|---|---|
| `narrow.count()` | **2** | 1 |
| `narrow.limit(5).collect()` | 1 | 1 |
| `wide.collect()` | **2** | 1 |

AQE를 끄면 액션 1개에 잡 1개로 딱 맞는다. 잡을 늘린 건 AQE다. 그런데 `limit(5).collect()`는 AQE를 켜도 잡이 1개다. 차이는 **셔플이 있느냐**다. 셔플이 있는 액션에서만 잡이 2개로 늘었다. 이유를 알려면 먼저 `count()`에 셔플이 왜 있는지부터 봐야 한다.

## 5. count()는 액션인데 왜 셔플이 생길까?

---

`narrow`에는 `filter`와 `withColumn`뿐이라 데이터 교환이 없다. 하지만 **`count()` 자체가 셔플을 만든다.**

파티션 4개가 각자 자기 행을 세는 건 따로 할 수 있다. 하지만 최종 답 "1000만"을 내려면 네 파티션의 숫자를 한곳에 모아 더해야 한다. 다른 파티션의 결과가 필요하니 데이터 교환이 생긴다. 그래서 스파크는 `count()`를 **파티션마다 미리 세기 → 한곳에 모아 더하기** 두 단계로 나눠 처리한다.

<div class="xd">
<div class="xd-dag">
<div class="xd-stage">
<div class="xd-stage-title">스테이지 1 · 태스크 4개</div>
<div class="xd-ops"><span class="xd-op">파티션 1: 250만</span><span class="xd-op">파티션 2: 250만</span><span class="xd-op">파티션 3: 250만</span><span class="xd-op">파티션 4: 250만</span></div>
<small>파티션마다 미리 셈</small>
</div>
<div class="xd-shuffle"><div>셔플</div><div class="xd-bar"></div><div class="xd-mono">1곳으로 모음</div></div>
<div class="xd-stage">
<div class="xd-stage-title">스테이지 2 · 태스크 1개</div>
<div class="xd-ops"><span class="xd-op">250만 × 4 = 1000만</span></div>
<small>파티션 1개로 모아서 더함</small>
</div>
</div>
</div>

1000만 행을 그대로 옮기는 대신 파티션마다 센 숫자 하나씩, 총 4개만 옮기니 가볍다. 그래도 데이터가 파티션 사이를 오가니 셔플이고, 스테이지가 2개로 나뉜다.

### 셔플이 있으면 AQE가 잡을 둘로 나눈다

셔플이 있는 액션에서 잡이 2개가 되는 이유는 AQE의 동작 방식에 있다.

1. 셔플 앞부분(스테이지 1)만 먼저 **잡 하나로 제출해서 실행**한다.
2. 셔플 결과의 실제 크기를 확인한다.
3. 그 통계로 뒷부분 계획을 다시 최적화한다.
4. 뒷부분(스테이지 2)을 **새 잡으로 제출**한다.

AQE를 끄면 잡 1개 안에 스테이지 2개가 들어간다. AQE를 켠 상태에서 잡과 스테이지를 세면 이렇다.

| 잡 | 스테이지 | 태스크 | 실제 실행된 태스크 |
|---|---|---|---|
| 잡 0 | 스테이지 0 (셔플 전) | 4 | 4 |
| 잡 1 | 스테이지 1 (셔플 전, 다시 표시) | 4 | **0 (skipped)** |
| 잡 1 | 스테이지 2 (셔플 후) | 1 | 1 |

잡 1에도 셔플 앞 스테이지가 들어 있지만, 잡 0에서 이미 끝났으니 셔플 결과를 재사용하고 건너뛴다. 스파크 UI에서 회색 "skipped" 스테이지로 보이는 게 이것이다.

`wide.collect()`도 같다. 스파크 UI에 스테이지가 3개로 보이지만 그중 1개는 skipped이고, **실제로 실행된 스테이지는 2개**다. 스테이지 ID 수와 실제 실행된 스테이지 수는 다를 수 있다.

## 6. 셔플 뒤 스테이지의 태스크는 몇 개일까?

---

스테이지 1은 파티션 4개라 태스크 4개다. 셔플 뒤 스테이지 2의 태스크 수는 AQE에 따라 다르다.

| `wide.collect()` | 스테이지 1 (셔플 전) | 스테이지 2 (셔플 후) |
|---|---|---|
| AQE 꺼짐 | 태스크 4 | 태스크 **200** |
| AQE 켜짐 | 태스크 4 | 태스크 **1** |

규칙은 그대로 **태스크 수 = 파티션 수**다. 달라지는 건 셔플 뒤의 파티션 수다.

- **AQE를 끄면** 셔플 뒤 파티션 수의 기본값(`spark.sql.shuffle.partitions`) 200을 그대로 쓴다. 그런데 `bucket` 값은 5개뿐이다. 데이터가 들어가는 파티션은 많아야 5개이고, **나머지 195개 이상은 빈 파티션을 처리하는 태스크**다.
- **AQE를 켜면** 셔플 앞부분을 먼저 실행한 뒤 결과 크기를 본다. 셔플 데이터는 파티션 4개 × bucket 5개, 숫자 20개 정도로 아주 작다. AQE는 작은 파티션들을 하나로 합쳐서(coalesce) 1개로 줄이고, 그래서 태스크도 1개가 된다.

## 7. 태스크 수는 어떻게 확인할까?

---

가장 쉬운 방법은 스파크 UI다. `pyspark` 셸을 켜 둔 채로 `http://localhost:4040`에 들어간다.

- **Jobs 탭**: 잡마다 완료/전체 태스크 수가 나온다.
- **Stages 탭**: 스테이지마다 `Tasks: Succeeded/Total`이 나온다. 5번 질문의 skipped 스테이지도 여기서 보인다.

실행 중에는 터미널에 나오는 진행 바로도 볼 수 있다.

```text
[Stage 0:=============>                    (2 + 2) / 4]
```

`(완료 + 실행 중) / 전체 태스크 수`라는 뜻이다. 위 예는 태스크 4개 중 2개가 끝났고 2개가 실행 중이다.

## 정리

---

- 잡은 **액션을 호출한 순서대로** 실행되고, **잡마다 자기 DAG**를 만든다. 셔플이 있으면 AQE 때문에 액션 하나에 잡이 2개 생길 수 있다.
- 스테이지는 **잡 하나 안에서 셔플 경계로** 나뉜다.
- **태스크 수 = 파티션 수**다. 코어 수는 태스크를 동시에 몇 개 실행하느냐만 정한다.
- 셔플 뒤 파티션은 기본 200개지만, AQE가 실제 크기를 보고 합친다. 이번 예제에서는 1개가 됐다.
