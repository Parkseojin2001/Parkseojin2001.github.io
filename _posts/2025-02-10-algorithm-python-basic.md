---
title: "Python 자료구조 함수"
description: "List 적용 함수 / Hash-Table 적용 함수"

categories: [Algorithm, Python]
tags: [python]

permalink: /python/basic/

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-02-10
last_modified_at: 2025-02-16
---

## List 관련 함수
---------

**sorted 함수 vs sorted 함수**<br>

`sorted(정렬할 데이터, key 파라미터, reverse 파라미터)`


sorted 함수는 파이썬 내장 함수이다.<br>
첫 번째 매개변수로 들어온 이터러블한 데이터를 **새로운 정렬된 리스트로 만들어서 반환**해 주는 함수이다.<br>

- 첫 번째 매개변수로 들어올 "정렬할 데이터"는 iterable(요소를 하나씩 차례로 반환 가능한 object) 한 데이터 이어야 합니다.

아래 옵션(파라미터)은 다 기본값으로 들어가 있기 때문에 sorted(정렬 데이터)만 넣어도 충분합니다.

- key 파라미터: sorted 함수의 key 파라미터는 어떤 것을 기준으로 정렬할 것인가? 에 대한 기준이다. 즉, key 값을 기준으로 비교를 하여 정렬을 하겠다는 것인데, 이것을 정해 줄 수 있는 파라미터이다. `sorted( ~~ , key=)`로 입력하게 되면 해당 키를 기준으로 정렬하여 반환한다.

- reverse 파라미터: 해당 파라미터를 이용하면 오름차순으로 정렬할지 내림차순으로 정렬할지 정할 수 있다. 디폴트로는 reverse=False로 오름차순으로 정렬이 된다.`sorted( ~~ , reverse=True)`로 입력하게 되면 내림차순으로 정렬하여 반환한다.

```python
# sorted() 정렬
a = [2, 4, 1, 9, 100, 29, 40, 10]
b = sorted(a)   # 오름차순 정렬
c = sorted(a, reverse=True)   # 내림차순 정렬
```


`List.sort()`

`List.sort(reverse 파라미터)`

list.sort() 메서드는 list 객체 자체를 정렬해주는 함수이며 리스트에만 사용이 가능하다. list 객체의 멤버 함수, 즉 메서드입니다.

list.sort() 함수는 기본적으로 리스트를 오름차순으로 정렬해주는 기능을 합니다.

```python
# sort() 정렬
list_num = [33, 2, 81, -77, 44, 1, 10, 99, 5, 0, -2]
list_num.sort()   # 오름차순 정렬
b.sort(reverse=False)   # 내림차순 정렬
```

>**sorted( ) vs sort( )**<br>
>새로운 정렬된 리스트를 반환하는 함수는 sorted 함수이고, 리스트 자체를 정렬시켜버리는 것은 sort 함수입니다.

**pop( ) vs popleft( )**<br>

`pop( )`은 동적 배열로 구성된 리스트에서 맨 뒤 아이템을 가져오는 건 적합하지만, 맨 앞 아이템을 가져오기 적합한 자료형이 아니다. 그 이유는 첫 번째 값을 꺼내오면 모든 값이 한 칸씩 시프팅되며, 시간 복잡도가 O(n)이 발생된다. 

`popleft( )`은 데크 자료형에 사용되는 함수이며 파이썬에서 데크는 이중 연결리스트로 구성되었기 때문에 시간 복잡도가 O(1)로 실행된다.

```python
q = collections.deque()
q.appendleft('e')   # 왼쪽에 데이터 삽입
q.append('r')     # 오른쪽에 데이터 삽입
q.popleft()   # 왼쪽 데이터 제거
q.pop()   # 오른쪽 데이터 제거
```

## Set 관련 함수
---------

`set.add(item)`
- item을 추가(중복되면 item 추가를 하지 않음)

`set.remove(item)`
- item에 해당하는 원소를 제거하고, 없으면 KeyError 발생

`set.discard(item)`
- item에 해당하는 원소를 제거하고, 없어도 에러발생하지 않음



## Hash Table 관련 함수
---------

**get() 함수의 작동 방식**<br>

`hash_table.get(key, default)`  
- key: 값을 찾고자 하는 키
  - 키가 사전에 있는 경우: 해당하는 값을 반환
  - 키가 사전에 없는 경우: default 값이 지정되어 있으면 해당 값을 반환하고 default 값이 생략되면 None을 반환
- default: 키가 사전에 없을 경우 반환할 기본값 (생략 가능)

```python 
my_dict = {"apple": 1, "banana": 2}

# 키가 사전에 있는 경우
value1 = my_dict.get("apple")  # 1 반환

# 키가 사전에 없는 경우
value2 = my_dict.get("grape", 0)  # 0 반환 (기본값 지정)
value3 = my_dict.get("grape")  # None 반환 (기본값 생략)
```