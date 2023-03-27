# Word2Vec & GloVe

Word2Vec과 GloVe는 모두 자연어 처리 분야에서 단어 임베딩을 학습하는 데 사용되는 인기있는 기술입니다.

#### Word2Vec
Word2Vec은 2013년에 Tomas Mikolov에 의해 개발된 알고리즘으로, 단어를 고정 길이 벡터로 표현하는 데 사용됩니다. Word2Vec은 CBOW (Continuous Bag-of-Words) 및 Skip-Gram 두 가지 모델을 사용합니다.

CBOW 모델에서는 주변 단어의 벡터들을 입력으로 사용하여 중심 단어를 예측합니다. Skip-Gram 모델에서는 중심 단어를 입력으로 사용하여 주변 단어를 예측합니다. Word2Vec은 모델을 학습하는 동안 단어의 분산 표현(distributed representation)을 학습하여 비슷한 의미를 가진 단어들이 비슷한 벡터로 표현되도록 합니다.
<!-- 
Word2Vec은 다음과 같은 수식으로 표현됩니다.

CBOW (Continuous Bag-of-Words) 모델의 수식은 다음과 같습니다.

$$\text{Predict}\left(w_t\right) = \frac{1}{2c} \sum_{-c\le j\le c, j\ne 0} \mathbf{v}_{t+j}$$

여기서 $w_t$는 중심 단어이고, $c$는 주변 단어의 수이며, $\mathbf{v}_{t+j}$는 $w_t$의 주변 단어 중 $j$번째 단어의 임베딩 벡터입니다.

Skip-gram 모델의 수식은 다음과 같습니다.

$$\text{Predict}\left(w_{t+j} \mid w_t\right) = \text{softmax}\left(\mathbf{v}{w{t+j}}^T \mathbf{v}_{w_t}\right)$$

여기서 $w_t$는 중심 단어이고, $w_{t+j}$는 $w_t$의 주변 단어 중 $j$번째 단어입니다. $\mathbf{v}{w_t}$와 $\mathbf{v}{w_{t+j}}$는 각각 $w_t$와 $w_{t+j}$의 임베딩 벡터입니다. $\text{softmax}$ 함수는 출력 벡터를 확률 분포로 변환합니다. -->

[ToDo:그림 삽입 예정]

#### GloVe(Global Vectors for Word Representation)
GloVe는 Word2Vec과 함께 가장 유명한 단어 임베딩 기법 중 하나입니다. Word2Vec과는 달리, GloVe는 전역 정보를 사용하여 단어 임베딩을 학습합니다.

GloVe는 기본적으로 다음과 같은 아이디어에 기반합니다. 어떤 텍스트 코퍼스에서 두 단어 $i$와 $j$가 함께 출현한 횟수가 많으면, 두 단어는 의미적으로 비슷하다고 할 수 있습니다. 이런 아이디어를 이용하여, GloVe는 전체 텍스트 코퍼스에서 두 단어가 함께 출현한 횟수의 로그를 임베딩 벡터 간의 내적으로 모델링합니다.

GloVe는 동시 등장 행렬(co-occurrence matrix)을 사용하여 단어의 의미를 파악합니다. 동시 등장 행렬은 언어 모델링 등 다양한 자연어 처리 작업에서 사용되는 행렬이며, 각 행과 열은 각각 단어의 빈도수를 나타냅니다. 예를 들어, 동시 등장 행렬에서 $(i, j)$ 번째 원소는 단어 $i$와 $j$가 함께 등장한 횟수를 나타냅니다.
<!-- 
GloVe 모델에서 두 단어 $i$와 $j$의 임베딩 벡터를 각각 $\mathbf{v}_i$와 $\mathbf{v}_j$라고 하면, 이들의 내적은 다음과 같이 표현할 수 있습니다.

$$\mathbf{v}_i^T \mathbf{v}j = \log \left( X{ij} \right)$$

여기서 $X_{ij}$는 $i$와 $j$가 함께 출현한 횟수입니다.

하지만 단순히 이런 내적만 사용하면, 비슷한 의미를 지닌 여러 단어에 대해서도 서로 다른 임베딩 벡터가 학습될 수 있습니다. 따라서 GloVe는 내적을 다음과 같이 수정하여 여러 단어를 함께 고려하도록 합니다.

$$\mathbf{v}_i^T \mathbf{v}j + b_i + b_j = \log \left( X{ij} \right)$$

여기서 $b_i$와 $b_j$는 각각 $i$와 $j$의 바이어스 값입니다. 이렇게 수정된 내적을 최소화하는 임베딩 벡터와 바이어스 값을 학습하면 됩니다.

GloVe 모델의 목적 함수는 다음과 같이 정의됩니다.

$$ J = \sum_{i=1}^{|V|} \sum_{j=1}^{|V|} f \left( X_{ij} \right) \left( \mathbf{v}_i^T \mathbf{v}j + b_i + b_j - \log \left( X{ij} \right) \right)^2 $$

여기서 $f \left( X_{ij} \right)$는 가중치 함수입니다. 이 함수는 $X_{ij}$의 값을 조정하여, 자주 출현하는 단어 쌍에 대해서는 작은 가중치를 부여하고, 드물게 출현하는 단어 쌍에 대해서는 큰 가중치를 부여합니다.

여기서 $|V|$는 어휘 집합의 크기입니다. $f \left( X_{ij} \right)$는 다음과 같이 정의됩니다.

$$ f \left( X_{ij} \right) = \begin{cases} \left( \frac{X_{ij}}{x_{\max}} \right)^\alpha & \text{if } X_{ij} < x_{\max} \ 1 & \text{otherwise} \end{cases} $$

여기서 $x_{\max}$는 임계값입니다. $X_{ij}$가 $x_{\max}$보다 작으면, $X_{ij}$의 값을 $\alpha$승한 값으로 가중치를 조정합니다. 이렇게 함으로써, 자주 출현하는 단어 쌍에 대해서는 작은 가중치를 부여하고, 드물게 출현하는 단어 쌍에 대해서는 큰 가중치를 부여할 수 있습니다. $\alpha$는 가중치 조정의 정도를 조절하는 하이퍼파라미터입니다.

GloVe 모델의 학습은 경사 하강법을 사용하여 수행됩니다. 최적화할 변수는 $\mathbf{v}_i$, $\mathbf{v}_j$, $b_i$, $b_j$ 입니다. 따라서 목적 함수 $J$를 $\mathbf{v}_i$, $\mathbf{v}_j$, $b_i$, $b_j$에 대해서 각각 편미분하여 경사 하강법으로 업데이트합니다. -->

GloVe 모델은 Word2Vec보다는 조금 느리지만, 더 나은 성능을 보일때도 있습니다. 또한, 단어 간의 선형적인 관계를 더 잘 표현할 수 있는 장점이 있습니다.


파이썬에서는 Gensim 라이브러리를 사용하여 Word2Vec과 GloVe 모델을 쉽게 구현할 수 있습니다. 다음은 예제 코드입니다.

[ToDo:코드 검수 예정]
```python
from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

# GloVe 파일을 Word2Vec 파일로 변환
glove_input_file = 'glove.6B.100d.txt'
word2vec_output_file = 'glove.6B.100d.word2vec.txt'
glove2word2vec(glove_input_file, word2vec_output_file)

# Word2Vec 모델 학습
model_w2v = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)

# GloVe 모델 로드
model_glove = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

# 단어 벡터 확인
print(model_w2v.wv['word'])
print(model_glove['word'])
```

위 코드에서 glove.6B.100d.txt는 GloVe 모델의 학습 데이터 파일이며, sentences는 학습에 사용될 문장 리스트입니다. Word2Vec 클래스를 사용하여 Word2Vec 모델을 학습하고, KeyedVectors.load_word2vec_format() 메서드를 사용하여 GloVe 모델을 로드합니다. 마지막으로 wv 속성을 사용하여 Word2Vec 모델에서 단어 벡터를 확인하고, KeyedVectors 객체를 통해 GloVe 모델에서 단어 벡터를 확인할 수 있습니다.
