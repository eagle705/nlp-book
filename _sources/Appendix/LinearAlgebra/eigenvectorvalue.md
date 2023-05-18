# Eigenvector와 Eigenvalue

Eigenvector(고유벡터)와 Eigenvalue(고유값)는 머신러닝에서 매우 중요한 개념입니다. 이들은 다양한 머신러닝 기법에서 사용되며 데이터 분석, 차원 축소, 이미지 처리, 패턴 인식 등에서 중요한 역할을 합니다. 

 <!-- Eigenvalue는 선형변환의 스케일링 계수를 나타내며, Eigenvector는 그 스케일링 계수를 곱해도 방향이 변하지 않는 벡터를 의미합니다. -->

수학적으로, A가 n x n 정사각 행렬이고 λ가 스칼라일 때, 만약 A에 대한 다음의 선형방정식이 성립한다면,

A * v = λ * v

여기서 v는 n차원 Eigenvector이고, λ는 그에 해당하는 Eigenvalue입니다. 이 식은 곧, A가 v 방향으로 상수배 λ만큼 변환하는 선형변환을 의미합니다.


## Eigenvector
**Eigenvector는** 선형 변환을 통해 벡터의 방향만 바뀌고 크기는 변하지 않는 벡터를 의미합니다. 이를 통해 데이터의 차원을 축소하거나, **주성분 분석(PCA)**을 수행하여 데이터의 주요 특성을 추출하는 등의 작업에 활용됩니다. 여기서 잠깐 PCA의 개념에 대해서 짧게 설명하면 다음과 같습니다.

### PCA
PCA(Principal Component Analysis)는 고차원 데이터의 차원을 줄이기 위한 기법으로, 고차원 데이터를 저차원 공간으로 변환하는 것이 핵심입니다. 이때, PCA는 데이터의 분산(variance)을 최대화하는 새로운 축(principal components)을 찾아내어 이를 이용하여 데이터를 변환합니다.

따라서, PCA를 수행하기 위해서는 먼저 데이터의 분산을 계산해야 합니다. 공분산(covariance)은 두 변수 간의 관계를 나타내는 통계량으로, 분산의 개념을 확장한 것입니다. 공분산 행렬(covariance matrix)은 여러 변수 간의 공분산을 포함하는 정방행렬입니다. 공분산 행렬을 이용하여 PCA를 수행하면, 각 변수 간의 상관관계를 고려하여 데이터를 변환하므로, 보다 정확하고 의미있는 결과를 얻을 수 있습니다.

따라서, PCA를 수행하기 전에 공분산 행렬을 계산하는 것은 중요합니다. 공분산 행렬 계산은 numpy의 cov 함수를 이용하여 간단히 수행할 수 있습니다. 다음은 numpy를 이용하여 공분산 행렬을 계산하는 예시 코드입니다.

```python
import numpy as np

# 데이터 생성
X = np.random.rand(100, 3)

# 공분산 행렬 계산
C = np.cov(X.T)
```

## Eigenvalue
**Eigenvalue**는 Eigenvector가 가지는 스케일링 상수로, Eigenvector를 변환한 후 벡터의 크기가 얼마나 커졌는지 또는 작아졌는지를 나타냅니다. 주어진 데이터셋에서 주요한 변동성(variability)의 양을 측정하는 지표입니다. 값이 클수록 해당 Eigenvector가 설명하는 변동성이 많아집니다. 이를 통해 **데이터의 중요도나 분산**을 계산할 수 있습니다. PCA(주성분분석)는 데이터를 고유벡터들의 선형 조합으로 표현하는 방식으로 데이터를 분해합니다.
이때, `고유값은 고유벡터가 얼마나 중요한지를 판단하는 기준`이 됩니다. 일반적으로 고유값이 큰 고유벡터는 주요한 변동성을 설명하는 특징을 담고 있으며, 작은 고유값은 노이즈나 덜 중요한 특징을 나타냅니다. 따라서, 고유값과 고유벡터는 데이터의 중요도를 판단하는데 활용됩니다. 특히, 고유값이 작은 고유벡터를 제거함으로써 데이터를 압축하고, 데이터 차원을 축소할 수 있습니다. 이를 통해, 데이터의 중요한 정보를 추출하면서도 더 적은 차원으로 데이터를 표현하는 것이 가능해집니다.

따라서, Eigenvector와 Eigenvalue는 머신러닝에서 매우 유용한 도구입니다. 이를 이해하고 활용하는 것은 머신러닝 이론을 깊이있게 이해하고, 머신러닝 모델을 설계하고 성능을 최적화하는 데에 필수적입니다.

예를 들어 , 이미지 데이터셋에서 얼굴을 인식하려고 한다고 가정해보겠습니다. 이때, 이미지 데이터는 벡터로 표현될 수 있으며, 여러 개의 이미지 데이터를 모아 하나의 행렬로 나타낼 수 있습니다. 이렇게 구성된 행렬은 대개 매우 큰 차원을 가지고 있습니다.

이때, Eigenvalue와 Eigenvector를 사용하여 데이터의 주요 특성을 추출할 수 있습니다. 예를 들어, 데이터의 분산이 큰 방향(가장 중요한 방향)을 나타내는 Eigenvector와 그 방향으로의 분산을 나타내는 Eigenvalue를 계산할 수 있습니다.

이렇게 계산된 Eigenvector와 Eigenvalue를 이용하여 PCA(Principal Component Analysis)를 수행하면, 데이터셋을 더 작은 차원으로 축소할 수 있습니다. 이렇게 축소된 데이터셋에서는, 얼굴 인식 알고리즘 등의 다양한 머신러닝 기법을 적용하여 얼굴을 인식할 수 있습니다.

또 다른 예로는 회귀 분석(Regression Analysis) 분야에서 Eigenvector와 Eigenvalue를 활용할 수 있습니다. 회귀 분석에서는 종속 변수와 독립 변수 사이의 관계를 모델링하는 것이 중요합니다. 이때, 독립 변수의 수가 매우 많은 경우, 다중 공선성(multicollinearity) 문제가 발생할 수 있습니다. 이 문제를 해결하기 위해 Eigenvalue와 Eigenvector를 사용하여 독립 변수를 축소하거나, 상호 작용 변수를 추출하는 등의 전처리 작업을 수행할 수 있습니다.

## Eigenvector와 Eigenvalue 구하는 조건
Eigenvector를 구할 수 있는 조건은 해당 행렬이 대칭행렬(Symmetric Matrix)이거나, 비대칭행렬(Asymmetric Matrix)이지만 complex eigenvalue가 없는 경우입니다.

대칭행렬(Symmetric Matrix)은 전치행렬(Transpose Matrix)과 원래 행렬이 같은 행렬을 말하며, 대표적으로 공분산행렬(Covariance Matrix)이 있습니다. 대칭행렬은 항상 Eigenvector를 가지며, Eigenvector들은 서로 직교(orthogonal)합니다.

비대칭행렬(Asymmetric Matrix)은 전치행렬과 원래 행렬이 다른 행렬을 말합니다. 이 경우, 행렬의 크기와 형태에 따라 Eigenvector가 존재하지 않을 수 있습니다. 또한, Eigenvector가 존재하더라도 complex eigenvalue가 있을 수 있습니다.

따라서, 일반적으로 머신러닝에서 사용되는 행렬은 대부분 대칭행렬 또는 비대칭행렬 중에서 대칭행렬인 경우가 많으며, 이 경우 항상 Eigenvector를 구할 수 있습니다.

예컨데 8x8 digits 데이터를 사용한다고 했을때, digits 데이터셋의 경우, 각 이미지 데이터의 픽셀 값을 feature로 사용하여 구성된 데이터 행렬 X가 존재합니다. 이 경우 X의 전치행렬 X^T와 X의 곱인 X^T * X를 사용하여 공분산 행렬(Covariance Matrix) C를 구할 수 있습니다.

공분산 행렬은 대칭행렬(Symmetric Matrix)이므로, 항상 Eigenvector를 가지게 됩니다. 따라서, digits 데이터셋에서도 Eigenvector를 구할 수 있었던 것입니다.

eigenvector를 그냥 계산하고 싶은건데 공분산 행렬을 꼭 구해야할까요?
Eigenvector를 구하기 위해서는 해당 행렬의 고유값(Eigenvalue)을 먼저 구해야 합니다. 이를 위해서는 해당 행렬을 구할 필요가 있습니다.

하지만, 공분산 행렬을 구하지 않고도 특이값 분해(Singular Value Decomposition, SVD)를 사용하여 Eigenvector를 구할 수 있습니다. SVD는 임의의 행렬에 대해서도 적용이 가능하며, 특히 대규모 데이터셋에서도 효과적으로 사용될 수 있습니다.

따라서, 공분산 행렬을 구하지 않고도 SVD를 사용하여 Eigenvector를 계산할 수 있습니다. 예를 들어, numpy 라이브러리의 linalg.svd 함수를 사용하면 됩니다.

```python
from sklearn.datasets import load_digits
import numpy as np

digits = load_digits()
X = digits.data

U, s, Vt = np.linalg.svd(X)

# calculate the principal components
pcs = Vt.T[:, :2]

# transform the data onto the first two principal components
transformed = np.dot(X, pcs)

```

위 예시에서는 digits 데이터셋의 특징(feature)을 X로 가져와 SVD를 사용하여 U, s, Vt를 구했습니다. 여기서 Vt.T[:, :2]는 첫번째와 두번째 Eigenvector를 추출하여 principal components를 계산한 것입니다.

SVD는 임의의 행렬에 대해서도 적용이 가능합니다. 따라서, digits 데이터셋 뿐만 아니라 다른 데이터셋에 대해서도 SVD를 사용하여 Eigenvector를 계산할 수 있습니다. 다만, 특정 데이터셋에 대해서는 공분산 행렬을 사용하는 것이 더 효율적일 수도 있습니다.

결국 머신러닝에서 Eigenvector와 Eigenvalue는 다양한 응용 사례가 있지만 데이터셋을 가장 잘 설명하는 중요한 축을 새로 뽑아서 차원을 축소하기 위해 주로 사용합니다.
