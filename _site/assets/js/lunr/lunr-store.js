var store = [{
        "title": "[포스팅 예시] 이곳에 제목을 입력하세요",
        "excerpt":"🦥 본문   본문은 여기에 …  ","categories": ["Operating System"],
        "tags": ["os"],
        "url": "/os/intro/",
        "teaser": null
      },{
        "title": "선형 자료구조(1)",
        "excerpt":"🦥 배열 배열(Array)은 값 또는 변수 엘리먼트의 집합으로 구성된 구조로, 하나 이상의 인덱스 또는 키로 식별된다. ADT의 실제 구현 대부분은 배열 또는 연결 리스트를 기반으로 한다. 배열은 크기를 지정하고 해당 크기만큼의 연속된 메모리 공간을 할당받는 작업을 수행하는 자료형을 말한다. 배열은 큐 구현에 사용되는 자료형이다. int arr[5] = {4, 7, 29,...","categories": ["Data Structure"],
        "tags": ["data-structure"],
        "url": "/data-structure/linear-1/",
        "teaser": null
      },{
        "title": "선형 자료구조(2)",
        "excerpt":"🦥 데크, 우선순위 큐 데크는 스택과 큐의 연산을 모두 갖고 있는 복합 자료형이며, 우선순위 큐는 추출 순서가 일정하게 정해져 있는 않은 자료형이다. 데크 데크(Deque)는 더블 엔디드 큐의 줄임말로, 글자 그대로 양쪽 끝을 모두 추출할 수 있는, 큐를 일반화한 형태의 추상 자료형(ADT)이다. 데크는 양쪽에서 삭제와 삽입을 모두 처리할 수 있으며, 스택과...","categories": ["Data Structure"],
        "tags": ["data-structure"],
        "url": "/data-structure/linear-2/",
        "teaser": null
      },{
        "title": "10장 케라스를 사용한 인공 신경망 소개(1)",
        "excerpt":"지능적인 기계를 만드는 법에 대한 영감을 얻으려면 뇌 구조를 살펴보는 것이 합리적이다. 이는 인공신경망(ANN; Artificial Neural Network)을 촉발시킨 근원이다. 인공신경망은 뇌에 있는 생물학적 뉴런의 네트워크에서 영감을 받은 머신러닝 모델이다. 하지만 최근 인공 신경망은 생물학적 뉴런에서 점점 멀어지고 있으며 이러한 특징을 반영하기 위해 뉴런을 대신해 유닛(unit)이라고 부른다. 10.1 생물학적 뉴런에서 인공...","categories": ["핸즈온 머신러닝"],
        "tags": ["hands-on"],
        "url": "/hands-on/ANN-1/",
        "teaser": null
      },{
        "title": "Python 자료구조 함수",
        "excerpt":"🦥 List 관련 함수 sorted 함수 vs sorted 함수 sorted(정렬할 데이터, key 파라미터, reverse 파라미터) sorted 함수는 파이썬 내장 함수이다. 첫 번째 매개변수로 들어온 이터러블한 데이터를 새로운 정렬된 리스트로 만들어서 반환해 주는 함수이다. 첫 번째 매개변수로 들어올 “정렬할 데이터”는 iterable(요소를 하나씩 차례로 반환 가능한 object) 한 데이터 이어야 합니다. 아래...","categories": ["Python"],
        "tags": ["python"],
        "url": "/python/basic/",
        "teaser": null
      },{
        "title": "정렬",
        "excerpt":"정렬 알고리즘은 목록의 요소를 특정 순서대로 넣는 알고리즘이다. 대개 숫자식 순서(Numerical Order)와 사전식 순서(Lexicographical Order)로 정렬한다. 🦥 버블 정렬 def bubbleSort(A): for i in range(1, len(A)): for j in range(0, len(A) - 1): if A[j] &gt; A[j + 1]: A[j], A[j + 1] = A[j + 1], A[j] 버블 정렬은...","categories": ["Algorithm"],
        "tags": ["algorithm"],
        "url": "/algorithm/sorting/",
        "teaser": null
      },{
        "title": "비선형 자료구조(1)",
        "excerpt":"🦥 그래프 수학에서, 좀 더 구체적으로 그래프 이론에서 그래프란 객체의 일부 쌍(pair)들이 ‘연관되어’ 있는 객체 집합 구조를 말한다. 300여 년 전 도시의 시민 한 명이 “이 7개 다리를 한 번씩만 건너서 모두 지나갈 수 있을까?”라는 흥미로운 문제를 냈으며 이를 오일러가 해법을 발견하는 것이 그래프 이론의 시작이다. 오일러 경로 아래 그림에서...","categories": ["Data Structure"],
        "tags": ["data-structure"],
        "url": "/data-structure/nonlinear-1/",
        "teaser": null
      },{
        "title": "Git 공부하기",
        "excerpt":"🦥 Git이란? Git은 Distributed Version Controll System(분산 버전 관리 시스템)으로 파일들을 추적하는 방식을 말한다. Git은 파일들의 모든 변경사항을 트래킹한다. 만약, 프로젝트를 git repository에 등록을 했다면 git은 등록된 모든 파일들을 추적한다. git은 파일을 binary code로 읽기때문에 원하는 것이 무엇이든지 다 읽을 수 있다. 🦥 Github란? Github는 작업한 git 파일(git 변경사항)을 업로드하는...","categories": ["Git"],
        "tags": ["git"],
        "url": "/git/basic/",
        "teaser": null
      },{
        "title": "비선형 자료구조(2)",
        "excerpt":"🦥 트리 트리는 계층형 트리 구조를 시뮬레이션하는 추상 자료형(ADT)으로, 루트 값과 부모-자식 관계의 서브트리로 구성되며, 서로 연결된 노드의 집합이다. 트리(Tree)는 하나의 뿌리에서 위로 뻗어 나가는 형상처럼 생겨서 트리(나무)라는 명칭이 붙었는데, 트리 구조를 표현할 때는 나무의 형상과 반대 방향으로 표현한다. 트리의 속성 중 하나는 재귀로 정의된 자기 참조 자료구조라는 점이다. 여러...","categories": ["Data Structure"],
        "tags": ["data-structure"],
        "url": "/data-structure/nonlinear-2/",
        "teaser": null
      },{
        "title": "10장 케라스를 사용한 인공 신경망 소개(2)",
        "excerpt":"10.2 케라스로 다층 퍼셉트론 구현하기 케라스는 모든 종류의 신경망을 손쉽게 만들고 훈련, 평가, 실행할 수 있는 고수준 딥러닝 API이다. 텐서플로와 케라스 다음으로 가장 인기 있는 딥러닝 라이브러리는 페이스북 파이토치 (PyTorch)이다. 10.2.1 텐서플로 2 설치 $ cd $ML_PATH # ML을 위한 작업 디렉토리 $ source my_env/bin/activate # 리눅스나 맥OS에서 $ .\\my_env\\Scripts\\activate...","categories": ["핸즈온 머신러닝"],
        "tags": ["hands-on"],
        "url": "/hands-on/ANN-2/",
        "teaser": null
      },{
        "title": "Python 문법",
        "excerpt":"🦥 zip() 함수 zip() 함수는 2개 이상의 시퀀스를 짧은 길이를 기준으로 일대일 대응하는 새로운 튜플 시퀀스를 만드는 역할을 한다. a = [1, 2, 3, 4, 5] b = [2, 3, 4, 5] c = [3, 4, 5] zip(a, b) # &lt;zip object at 0x105b6d9b0&gt; 파이썬 3+에서는 제너레이터를 리턴한다. 제너레이터에서 실제값을...","categories": ["Python"],
        "tags": ["python"],
        "url": "/python/grammar/",
        "teaser": null
      },{
        "title": "10장 케라스를 사용한 인공 신경망 소개(3)",
        "excerpt":"10.3 신경망 하이퍼파라미터 튜닝하기 신경망에는 조정할 하이퍼파라미터가 많다. 최적의 하이퍼파라미터 조합을 찾는 방식에는 검증 세트에서 (또는 K-fold 교차 검증으로) 가장 좋은 점수를 내는지 확인하는 것이다. GridSearchCV나 RandomizedSearchCV를 사용해 하이퍼파라미터 공간을 탐색할 수 있다. def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]): model = keras.models.Sequential() model.add(keras.layers.InputLayer(input_shape=input_shape)) # 입력 크기 for layer in range(n_hidden): #...","categories": ["핸즈온 머신러닝"],
        "tags": ["hands-on"],
        "url": "/hands-on/ANN-3/",
        "teaser": null
      },{
        "title": "11장 케라스를 사용한 인공 신경망 소개(1)",
        "excerpt":"데이터에 따라 깊은 심층 신경망을 훈련해야 한다. 심층 신경망을 훈련하는 도중에 다음과 같은 문제를 마주할 수 있다. 그레이디언트 소실 또는 그레이디언트 폭주 문제에 직면할 수 있다. 심층 신경망의 아래쪽으로 갈수록 그레이디언트가 점점 더 작아지거나 커지는 현상이다. 두 현상 모두 하위층을 훈련하기 매우 어렵게 만든다. 대규모 신경망을 위한 훈련 데이터가 충분하지...","categories": ["핸즈온 머신러닝"],
        "tags": ["hands-on"],
        "url": "/hands-on/DNN-1/",
        "teaser": null
      },{
        "title": "11장 케라스를 사용한 인공 신경망 소개(2)",
        "excerpt":"11.3 고속 옵티마이저 훈련 속도를 높이는 방법 연결 가중치에 좋은 초기화 전략 사용하기 좋은 활성화 함수 사용하기 배치 정규화 사용하기 사전훈련 네트워크 일부 재사용 훈련 속도를 크게 높일 수 있는 또 다른 방법으로 표준적인 경사 하강법 옵티마이저 대신 더 빠른 옵티마이저를 사용할 수 있다. 모멘텀 최적화 네스테로프(Nesterov) 가속 경사(NAG) AdaGrad...","categories": ["핸즈온 머신러닝"],
        "tags": ["hands-on"],
        "url": "/hands-on/DNN-2/",
        "teaser": null
      },{
        "title": "11장 케라스를 사용한 인공 신경망 소개(3)",
        "excerpt":"11.4 규제를 사용해 과대적합 피하기 심층 신경망의 높은 자유도는 네트워크를 훈련 세트에 과대적합되기 쉽게 만들기 때문에 규제가 필요하다. 조기종료 기법: EarlyStopping 콜백을 사용하여 일정 에포크 동안 성능이 향상되지 않는 경우 자동 종료시키기 배치 정규화: 불안정한 그레이디언트 문제해결을 위해 사용하지만 규제용으로도 활용 가능(가중치 변화를 조절하는 역할) 11.4.1 $\\ l_1$ 과 $\\...","categories": ["핸즈온 머신러닝"],
        "tags": ["hands-on"],
        "url": "/hands-on/DNN-3/",
        "teaser": null
      },{
        "title": "12장 텐서플로를 사용한 사용자 정의 모델과 훈련",
        "excerpt":"12.1 텐서플로 훑어보기 텐서플로는 강력한 수치 계산용 라이브러리이다. 특히 대규모 머신러닝에 잘 맞도록 튜닝되어 있다. 핵심 구조는 넘파이와 매우 비슷하지만 GPU를 지원한다. 분산 컴퓨팅을 지원한다. 일종의 JIT 컴파일러를 포함한다. 속도를 높이고 메모리 사용량을 줄이기 위해 계산을 최적화한다. 이를 위해 파이썬 함수에서 계산 그래프를 추출한 다음 최적화하고 효율적으로 실행한다. 계산 그래프는...","categories": ["핸즈온 머신러닝"],
        "tags": ["hands-on"],
        "url": "/hands-on/tf-1/",
        "teaser": null
      },{
        "title": "MLflow 소개 및 Tutorial",
        "excerpt":"머신러닝 프로세스의 관리할 수 있는 오픈소스인 MLflow에 대한 소개 및 간단한 Tutorial에 대한 글입니다. 🦥 MLflow MLflow는 End to End로 머신러닝 라이프 사이클을 관리할 수 있는 오픈소스 주요 기능 1) MLflow Tracking 모델에 대한 훈련 통계(손실, 정확도 등) 및 하이퍼 매개변수를 기록 나중에 검색할 수 있도록 모델을 기록(저장)한다. MLflow 모델...","categories": ["Capstone Design"],
        "tags": ["capstone-design"],
        "url": "/capstone-design/mlflow/",
        "teaser": null
      },{
        "title": "토큰화",
        "excerpt":"자연어 처리(NLP)는 컴퓨터가 인간의 언어를 이해하고 해석 및 생성하기 위한 기술을 의미한다. 자연어 처리는 인공지능의 하위 분야 중 하나로 컴퓨터가 인간과 유사한 방식으로 인간의 언어를 이해하고 처리하는 것이 주요 목표 중 하나다. 인간 언어의 구조, 의미, 맥락을 분석하고 이해할 수 있는 알고리즘과 모델을 개발한다. 이런 모델을 개발하기 위해서는 해결해야할 문제가...","categories": ["NLP"],
        "tags": ["NLP"],
        "url": "/nlp/tokenization/",
        "teaser": null
      },{
        "title": "파이토치 설치",
        "excerpt":"🦥 파이토치란? 파이토치(PyTorch)는 딥러닝 및 인공지능 애플리케이션에 널리 사용되는 파이썬용 오픈 소스 머신러닝 라이브러리다. 🦥 파이토치 특징 파이토치의 주요 기능은 다음과 같다. 동적 계산 그래프 GPU 가속 사용하기 쉬움 우수한 성능 활발한 커뮤니티 몇 가지 제한사항과 잠재적인 단점이 있다. 제한된 프로덕션 지원 제한된 문서 호환성 제한된 통합 🦥 파이토치 설치...","categories": ["Pytorch"],
        "tags": ["pytorch"],
        "url": "/pytorch/installation/",
        "teaser": null
      },{
        "title": "파이토치 기초(1)",
        "excerpt":"🦥 텐서 텐서(Tensor)란 넘파이 라이브러리의 ndarray 클래스와 유사한 구조로 배열(Array)이나 행렬(Matrix)과 유사한 자료 구조(자료형)다. 파이토치에서는 텐서를 사용하여 모델의 입출력뿐만 아니라 모델의 매개변수를 부호화(Encode)하고 GPU를 활용해 연산을 가속화할 수 있다. 넘파이와 파이토치 공통점: 수학 계산, 선형 대수 연산을 비롯해 전치(Tranposing), 인덱싱(Indexing), 슬라이싱(Slicing), 임의 샘플링(Random Sampling) 등 다양한 텐서 연산을 진행할 수...","categories": ["Pytorch"],
        "tags": ["Pytorch"],
        "url": "/pytorch/basic-1/",
        "teaser": null
      },{
        "title": "임베딩(1)",
        "excerpt":"컴퓨터는 텍스트 자체를 이해할 수 없으므로 텍스트를 숫자로 변환하는 텍스트 벡터화(Text Vectorization) 과정이 필요하다. 텍스트 벡터화란 텍스트를 숫자로 변환하는 과정을 의미한다. 기초적인 텍스트 벡터화로는 원-핫 인코딩(One-Hot Encoding), 빈도 벡터화(Count Vectorization) 등이 있다. 원-핫 인코딩: 문서에 등장하는 각 단어를 고유한 색인 값으로 매핑한 후, 해당 색인 위치를 1로 표시하고 나머지 위치를...","categories": ["NLP"],
        "tags": ["NLP"],
        "url": "/nlp/embedding-1/",
        "teaser": null
      },{
        "title": "AWS Cloud Introduction",
        "excerpt":" ","categories": ["AWS SAA"],
        "tags": ["AWS SAA"],
        "url": "/aws_saa/intro/",
        "teaser": null
      },{
        "title": "임베딩(2)",
        "excerpt":"🦥 Word2Vec Word2Vec은 단어 간의 유사성을 측정하기 위해 분포 가설(distributional hypothesis)을 기반으로 개발된 임베딩 모델이다. 분포 가설: 같은 문맥에서 함께 자주 나타나는 단어들은 서로 유사한 의미를 가질 가능성이 높다는 가정이며 단어 간의 동시 발생(co-occurrence) 확률 분포를 이용해 단어 간의 유사성을 측정 ex. ‘내일 자동차를 타고 부산에 간다’ 와 ‘내일 비행기를...","categories": ["NLP"],
        "tags": ["NLP"],
        "url": "/nlp/embedding-2/",
        "teaser": null
      }]
