---
title: "Numpy 라이브러리 사용법"
description: "네이버 부스트코스의 Pre-course 강의를 기반으로 작성한 포스트입니다."

categories: [Naver-Boostcamp, Pre-Course 1]
tags: [Naver-Boostcamp, Pre-Course, python, numpy]

permalink: /boostcamp/pre-course/numpy/

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-07-21
last_modified_at: 2025-07-21
---

## Numpy
-----------

Python에서 행렬식 계산을 어떻게 구현할 수 있을까? Python의 기본 이차원 리스트를 사용할 수 있지 않을까?

- 문제점
    - 방정식 계산 등을 위해서는 계산 기능을 직접 구현해야 한다.
    - List는 메모리 주소를 연결하는 형태기 때문에 굉장히 큰 행렬을 표현하는것이 다소 어렵다.
    - 인터프리터 언어라 처리 속도가 느리다.

이러한 점들을 해결하기 위해 만들어진 Python용 고성능 과학계산용 패키지가 `Numpy`이다.

Matrix와 Vector와 같은 Array 연산의 사실상 표준 역할을 하고 있다.

`Numpy`의 **명확한 장점**은 다음과 같다.

- 일반 List에 비해 빠르고, 효율적이다.
- 반복문 없이 데이터 배열 처리를 지원한다.
- 선형대수와 관련된 다양한 기능을 제공한다.
- `C`, `C++`, `Fortran` 등의 언어와 통합이 가능하다.

### ndarray 기본

numpy는`import numpy as np`로 호출하면 된다.

일반적으로 numpy는 `np`라는 alias(별칭) 이용해서 호출한다.

#### array 생성

`np.array` 함수를 활용하여 ndarray 배열을 생성한다. 일반적인 Python List와 달리, 이런 차이점을 가진다.

- **하나의 데이터 type**만 배열에 넣을 수 있다.
- **dynamic typing이 지원되지 않는다.** 따라서, array 생성 시점에서 타입을 지정해주어야 한다.
- `C`의 Array를 사용하여 배열을 생성한다.

```python
import numpy as np

test_array = np.array([1, 4, 5, 8], float)
test_array  # array([1., 4., 5., 8.])
type(test_array)    # ndarray
type(test_array[3]) # numpy.float64
```

#### `Numpy`와 List의 array 생성방식

- **`Numpy` ndarray**
    - 일반적인 array의 형태와 같이, 데이터들의 하나의 주소 영역에 몰려서 붙어있는 형태이다.
    - List에 비해 연산 효율성이 높다.
    - 메모리 크기가 일정하기 때문에 데이터 공간을 잡기에도 효율적이다.

- **Python List**
    - -5 ~256까지의 자주 사용하는 값들은 특정 주소(static한 위치)에 저장되어있다.
    - 따라서 해당 숫자들을 할당하면, value 부분에 해당 값들의 static한 주소가 할당된다.
    - **변수 &rarr; 주소 값 &rarr; 실제 값**으로 연결되는 형태로, 주소 값을 바꾸어주면 내부 원소들의 변경이 아주 쉬워진다.
    - 이것이 Python List가 다른 언어에 비해 변형이 용이한 이유
    - 그러나 여러번의 주소를 타고가야 하는 이런 형태 때문에, array에 비해 연산 효율성이 낮다.
    - 또, dynamic typing을 지원하므로 메모리 크기를 지정하기 어려워 공간 측면에서도 비효율적이다.

<img src="http://jakevdp.github.io/images/array_vs_list.png">

```python
a = [1, 2, 3, 4, 5]
b = [5, 4, 3, 2, 1]
a[0] is b[-1]   # True
# is는 주소값을 비교하는 연산, 1의 주소는 static해서 같음

a = np.array(a)
b = np.array(b)
a[0] is b[-1] # False
# List와 달리 numpy는 array 형태이므로, 두 ndarray의 원소 주소는 다르다.
```

#### array 정보 관련 코드

- `shape` : ndarray의 dimension 구성을 반환한다.
    - ex. 3rd order tensor : (Channel, Height, Width)
- `dtype` : ndarray의 데이터 type을 반환한다.
    - type은 Python뿐만 아니라 `C`의 data type도 사용할 수 있다.
- `ndim` : dimension 수준(rank)를 반환한다 - 쉽게 말해 괄호 level, 차원과 같다.
- `size` : data의 개수를 반환한다.
- `nbyte` : ndarray object의 메모리 크기를 반환한다.

**<Arrary Rank에 따라 불리는 이름>**

|**Rank**|**Name**|**Example**|
|--------|--------|-----------|
|0|scalar|`7`|
|1|vector|`[10, 10]`|
|2|matrix|`[[10, 10,], [15, 15]]`|
|3|3-tensor|`[[[1,5,9], [2,6,10]], [[3,7,11], [4,8,12]]]`|
|4|n-tensor||

```python
tensor = [[[1,2,5,8],[1,2,5,8],[1,2,5,8]],
       [[1,2,5,8],[1,2,5,8],[1,2,5,8]],
       [[1,2,5,8],[1,2,5,8],[1,2,5,8]],
       [[1,2,5,8],[1,2,5,8],[1,2,5,8]]]
tensor = np.array(tensor, int)

tensor.shape # (4, 3, 4)
tensor.dtype # dtype('int64')
tensor.ndim # 3
tensor.size # 48
tensor.nbytes # 384
```

### Handling shape

#### reshape

Array의 shape을 변경한다. element 개수는 동일하게 유지한다.

<img src="https://files.realpython.com/media/numpy-reshape-24-42.679add5e8d11.png">

`reshape`는 원본을 변경하지 않고 변경된 matrix를 반환한다.

```python
test_matrix = [[1, 2, 3, 4], [1, 2, 5, 8]]
np.array(test_matrix).shape # (2, 4)
np.array(test_matrix).reshape(8,).shape # (8, )

# -1을 지정해주면, size를 기반으로 다른 값들을 보고 해당 값을 선정한다.
np.array(test_matrix).reshape(-1,2).shape # (4,2)
```

#### flatten

다차원 array를 1차원 array로 변환한다.

원본을 변경하지 않고, 변경된 matrix를 반환한다.

```python
test_matrix = [[[1, 2, 3, 4], [1, 2, 5, 8], [1, 2, 3, 4], [1, 2, 5, 8]]]
np.array(test_matrix).flatten()
# array([1, 2, 3, 4, 1, 2, 5, 8, 1, 2, 3, 4, 1, 2, 5, 8])
```

### 인덱싱과 슬라이싱

#### indexing

리스트와 달리, 2차원 배열에서 [0, 0] 표기법을 제공한다.

matrix일 경우 앞은 row, 뒤는 column을 의미한다.

```python
test_example = np.array([[1, 2, 3], [4, 5, 6]], int)
test_example[0][2] # 3
test_example[0, 2] # 3
```

#### slicing

리스트와 달리, **행과 열 부분을 나누어서 slicing이 가능**하다.

matrix의 부분집합을 추출할 때에 유용하다.

```python
a = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], int)
a[:,2:] # 전체 Row의 2열 이상 -> array([[3, 4, 5], [8, 9, 10]])
a[1, 1,:3]  # 1 Row의 1열~2열 -> array([7, 8])
a[1:3]  # 1 Row ~ 2 Row 전체 -> array([[6, 7, 8, 9, 10]])
```

### Create Function

#### arange

array의 범위를 지정하여 값의 리스트를 생성한다.

```python
np.arange(30) # 0~29까지의 array
np.arange(0, 5, 0.5) # arange(시작, 끝, step)
np.arange(30).reshape(5,6) # 5 x 6 형태의 matrix 생성
```

#### ones, zeros, empty

- `ones`: 1로 가득찬 ndarray를 생성한다.

```python
np.ones(shape=(10,), dtype=np.int8) # array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int8)
np.ones((2,5)) # array([[1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.]])
```

- `zeros`: 0로 가득찬 ndarray를 생성하거나, 빈 ndarray를 생성한다.

```python
np.zeros(shape=(10,), dtype=np.int8) # array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int8)
np.zeros((2,5)) # array([[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]])
```

- `empty`: shape만 주어지고 비어있는 ndarray 생성(memory initialization이 되지 않음)

```python
np.empty(shape=(5,), dtype=np.int8) # array([0, 0, 0, 64, -74], dtype=int8)
np.empty((2, 4))
"""
array([[4.9e-324, 9.9e-324, 1.5e-323, 2.0e-323],
       [4.9e-324, 9.9e-324, 2.5e-323, 4.0e-323]])
"""
```

#### something_like

기존 ndarray shape 크기만큼 1, 0 또는 empty array를 반환한다.

```python
test_matrix = np.arange(30).reshape(5, 6)
np.ones_like(test_matrix)
"""
array([[1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1]])
"""
```

#### identity

단위 행렬(i 행렬)을 생성한다.

```python
np.identity(n=3, dtype=np.int8) # n == row 개수 == column 개수
"""
array([[1, 0, 0],
       [0, 1, 0],
       [0, 0, 1]], dtype=int8)
"""
```

#### eye

대각선이 1인 행렬을 생성한다.

`identity` 함수와 다른 점은, k값을 파라미터로 넘겨 시작 index의 변경이 가능하다는 것이다. 또 행과 열의 크기를 다르게 지정 가능하다.

```python
np.eye(3)
"""
array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])
"""

np.eye(N=3, M=5, dtype=np.int8)
"""
array([[1, 0, 0, 0, 0],
       [0, 1, 0, 0, 0],
       [0, 0, 1, 0, 0]], dtype=int8)
"""

np.eye(3, 5, k=2)
"""
array([[0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]])
"""
```

#### diag

대각행렬의 값을 추출한다. k값을 지정하여 시작 index를 바꿀수 있다.

```python
matrix = np.arange(9).reshape(3, 3)
np.diag(matrix) # array([0, 4, 8])
np.diag(matrix, k=1) # array([1, 5])
```

#### random sampling

데이터 분포에 따른 sampling으로 array를 생성한다.

```python
# random.distribution_type(start, end, size)
np.random.uniform(0, 1, 10).reshape(2, 5)   # 균등분포
np.random.normal(0, 1, 6).reshape(3, 2)    # 정규 분포
np.random.exponential(scale=2, size=100)  # 지수 분포
```

### Operation functions

#### sum

- `sum` : ndarray의 element 합을 구한다. 리스트의 `sum` 기능과 같다.

```python
test_array = np.arange(1, 11)
test_array # array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
test_array.sum(dtype=np.float) # 55.0
```

#### axis

모든 operation function을 실행할 때 기준이 되는 dimension 축을 의미한다.

`shape`를 찍었을 때, 앞의 숫자부터 axis index로 볼 수 있다.

```python
test_array = np.arange(1, 13).reshape(3, 4)
test_array
"""
array([[1, 2, 3, 4],
       [5, 6, 7, 8],
       [9, 10, 11, 12]])
axis=0은 column을 기준으로
axis=1은 row를 기준으로
"""
test_array.sum(axis=1) # array([10, 26, 42])
test_array.sum(axis=0) # array([15, 18, 21, 24])
```

- 축을 추가하는 방법
    - `reshape` : 파라미터로 -1을 넘기고, 나머지 축을 size 크기로 맞춰주기
    - `newaxis` : 파라미터에 `np.newaxis`를 추가하고 나머지 축에 `:` 사용

#### mean, std

- `mean`
    - element 평균을 반환한다.
- `std`
    - element 표준편차를 반환한다.

```python
test_array = np.arange(1,13).reshape(3,4)
test_array.sum(axis=1) # 각 row마다 합을 계산하여 array 반환
test_array.mean(axis=0) # 각 column마다 평균을 계산하여 array 반환
test_array.std(axis=0) # 각 column마다 표준편차를 계산하여 array 반환
```

이외에도 **mathematical function**의 종류가 다양하다.

`np.something`을 이용하여 종류를 확인할 수 있다.

- **exponential**
    - exp, expml, exp2, log, log10, log1p, log2, power, sqrt
- **trigonometric**
    - sin, cos, tan, acsin, arccos, atctan
- **hyperbolic**
    - sinh, cosh, tanh, acsinh, arccosh, atctanh

#### concatenate

numpy array를 합쳐주는 함수으 종류는 다음과 같다.

- `vstack`
    - 두 array를 수직(vertical)으로 붙인 새로운 matrix를 반환한다.
- `hstack`
    - 두 array를 수평(horizon)으로 붙인 새로운 matrix를 반환한다.

<img src="../assets/img/post/naver-boostcamp/numpy_stack.png">

- `concatenate((m1,m2), axis=n)`
    - 두 array를 지정한 axis에 따라 붙인 새로운 matrix를 반환한다.

<img src="https://content-media-cdn.codefinity.com/courses/4f4826d5-e2f8-4ffd-9fd0-6f513353d70a/section_3/concatenation_2d.png">

```python
a = np.array([1,2,3])
b = np.array([2,3,4])
np.vstack((a,b)) # 수직 병합
"""
array([[1, 2, 3],
       [2, 3, 4]])
"""

np.hstack((a,b)) # 수평 병합
"""
array([1, 2, 3, 2, 3, 4])
"""
```

```python
a = np.array([[1,2],[3,4]])
b = np.array([[5, 6]])
np.concatenate((a,b.T), axis=1) # column 방향으로 붙임
"""
array([[1, 2, 5],
       [3, 4, 6]])
"""

np.concatenate((a, b), axis=0)  # row 방향으로 붙임
"""
array([[1, 2],
       [3, 4],
       [5, 6]])
"""
```

### array operations

#### Operations b/t arrays

`Numpy`는 array 간의 기본적인 사칙연산을 지원한다.

단, **Element-wise Operation, 즉 Array간의 shape가 같을 때만 지원한다.** 그래야만 같은 위치에 있는 원소끼리 계산할 수 있기 때문이다.

#### Dot product

행렬 간 곱셈 연산으로, `dot` 함수를 사용한다.

```python
test_a = np.arange(1, 7).reshape(2, 3)
test_b = np.arange(7, 13).reshape(3, 2)
test_a.dot(test_b)
"""
array([[58, 64],
       [139, 154]])
```

<img src="https://i.ytimg.com/vi/uS3Ytql-efk/maxresdefault.jpg">

#### transpose

행렬을 전치시키는 연산이다. `transpose()` 또는 `T` 어트리뷰트를 사용한다.

`Pytorch` 모델을 작성하거나, 논문 구현 시에 자주 사용하는 operation이다.

```python
test = np.arange(1, 7).reshape(2, 3)
test.transpose()
test.T
"""
array([[1, 4],
       [2, 5],
       [3, 6]])
"""
```

#### broadcasting

**shape이 서로 다른 배열 간의 연산**을 지원하는 기능이다.

Scalar - vector 외에 vector - matrix 간 연산도 지원한다.

<img src="https://i.sstatic.net/JcKv1.png">

```python
test_matrix = np.array([[1, 2, 3], [4, 5, 6]], float)
scalar = 3

test_matrix + scalar # broadcasting 연산
"""
array([[4., 5., 6.],
       [7., 8., 9.]])
"""
```

### Numpy performance

- `%timeit [함수명 or 코드]` : `Jupyter` 환경에서 코드 퍼포먼스를 체크하는 함수
- `Numpy`는 C로 구현되어 있어, 성능을 확보하는 대신에 Python의 dynamic typing이라는 큰 특징을 희생했다.
- 일반적으로 연산속도는 `for loop` < `list comprehension` < `Numpy` 순으로 빠르다.
    - 1억번의 루프 기준으로 **약 4배 이상의 성능차이**를 보인다.
- 대용량 계산에서는 `Numpy` 가 가장 흔히 사용된다.
- `Concatenate` 처럼 계산이 아닌 **할당에서는 연산속도의 이점이 없다.**

### Comparison

#### All & Any

Array의 데이터 전부(and) 또는 일부(or)가 조건에 맞는지 확인하고, 만족 여부를 반환한다.

```python
a = np.arange(10)
np.any(a>5), np.any(a<0) # (True, False)
np.all(a>5), np.all(a<10) # (False, True)
```

#### Comparison operation

- 배열의 크기가 동일할때 element-wise operation으로 element간 비교 결과를 Boolean type 반환한다.
- `logical_and`, `logical_or`, `logical_not` 등의 연산을 이용하여 각 조건들이나 boolean 상태를 조합할 수 있다.

```python
test_a = np.array([1,2,3], int)
test_b = np.array([1,2,4], int)
test_a == test_b # array([True, True, False])
(test_a > test_b).any() # True

a = np.array([1, 3, 0], float)
np.logical_and(a > 0, a < 3)  # array([True, False, False], dtype=bool)
```

#### np.where

여러가지 사용법을 가진다.

- `where(condition, True, False)`
    - condition의 만족여부에 따라서 세팅해둔 값을 집어넣은 array를 반환한다.
- `where(condition)`
    - condition을 만족하는 원소들의 index값 array를 반환한다.

```python
a = np.array([3, 5, -1])
np.where(a > 0, 3, 2)  # array([3, 3, 2])

b = np.arange(10)
np.where(a>7) # array([8, 9])
```

다음과 같이 특이한 값, 함수들을 이용할 수도 있다.

- `np.Nan`, `np.Inf`
- `np.isnan()`, `np.isfinite()`

```python
a = np.array([1, np.NaN, np.Inf], float)  # NaN = Not a Number, Inf = Infinite
np.isnan(a)  # array([False, True, False], dtype=bool)
np.isfinite(a)  # array([True, False, False], dtype=bool)
```

#### argmax & argmin

Array 내부의 최대값 또는 최소값의 index를 반환한다.

axis를 지정해 각 axis별로 계산한 값을 반환받을 수도 있다.

```python
a = np.array([[1, 2, 4, 7], [9, 88, 6, 45], [9, 76, 3, 4]])
np.argmax(a)  # 5 --> flatten한 array 기준으로 index를 뽑는다
np.argmax(a, axis=1)    # array([3, 1, 1])
np.argmin(a, axis=0)    # array([0, 0, 2, 2])

np.argsort(a) # a를 sorting할 수 있다
```

### boolean & fancy index

#### boolean index

특정 조건의 여부 값을 배열 형태로 추출하여 사용한다.

Comparison operation 함수들도 모두 사용가능하다.

```python
test_array = np.array([1,4,0,2,3,8,9,7], float)
test_array > 3
"""
array([False,  True, False, False, False,  True,  True,  True])
"""

# 이를 이용하여 조건을 만족하는 element들만 추출할 수도 있다.
test_array[test_array > 3] # 조건이 True인 index의 element만 추출
"""
array([4., 8., 9., 7.])
"""

# 아래와 같은 형태로 자주 사용한다.
condition = test_array < 3
test_array[condition]
"""
array([1., 0., 2.])
"""
```

#### fancy index

`Numpy`는 array를 index value로 사용해서 값을 추출하므로, 특정 array를 다른 array의 인덱스에 집어넣어 value값들을 추출해 낼 수도 있다.

matrix 형태의 데이터에서도 추출 가능하다.

```python
a = np.array([1, 2, 3, 4, 5], float)
b = np.array([0, 0, 3, 1, 2], int)  # index 값이므로 반드시 int로 선언해야 한다
a[b]
"""
array([1., 1., 4., 2., 3.])
"""

a.take(b) # take 함수: bracket index와 같은 효과를 가진다.
"""
array([1., 1., 4., 2., 3.])
"""
```

### Numpy Data I/O

#### loadtxt & savetxt

text 타입 데이터를 읽고, `Numpy` object를 저장할 수 있다.
- `save`만 사용하면 `pickle` 형태로 저장된다.

```python
a = np.loadtxt("./sample.txt")
a[:10]  # 10개 데이터 가져오기
a_int = a.astype(int)  # integer 변환
a_int_3 = a_int[:3]
np.savetxt("int_data_3.csv", a_int_3, fmt="%.2", delimeter=",")
```


