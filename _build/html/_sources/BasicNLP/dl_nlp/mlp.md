# MLP(Multi-Layer Perceptron)

MLP(Multi-Layer Perceptron)은 딥러닝 모델 중 가장 기본적인 형태 중 하나로, 입력층(input layer), 은닉층(hidden layer), 출력층(output layer)으로 구성된 인공 신경망 구조입니다.

각각의 층은 뉴런들로 구성되어 있으며, 입력층은 입력 데이터를 받아들이고, 은닉층은 입력 데이터를 처리하여 중간 출력값을 계산하고, 출력층은 최종 출력값을 계산하여 결과를 제공합니다.

입력값은 일반적으로 벡터 형태로 제공되며, MLP는 입력값에 대한 가중치(weight)와 편향(bias)을 곱한 후 활성화 함수(activation function)를 적용하여 각 뉴런의 출력값을 계산합니다.

MLP의 학습은 역전파(backpropagation) 알고리즘을 사용하여 이루어지며, 학습 데이터를 이용하여 가중치와 편향을 업데이트하면서 모델의 오류를 최소화하도록 합니다.

아래는 간단한 MLP 모델을 구현한 코드 예시입니다.

```python
import torch
import torch.nn as nn

# MLP 모델 정의
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 모델 생성
input_dim = 10
hidden_dim = 20
output_dim = 1
model = MLP(input_dim, hidden_dim, output_dim)

# 입력 데이터 생성
x = torch.randn(32, input_dim)

# 모델에 입력 데이터를 전달하여 출력값 계산
y = model(x)
print(y.shape)

```

위 코드는 입력 차원이 10, 은닉층 크기가 20, 출력 차원이 1인 MLP 모델을 정의하고, 입력 데이터로 32개의 벡터를 생성하여 모델에 전달한 후, 출력값의 형태를 출력하는 예시입니다.

## Universal Approximation Theorem
Universal Approximation Theorem(전체근사정리)은 딥러닝과 같은 신경망이 어떠한 연속 함수든지 근사할 수 있다는 정리입니다. 이 정리는 딥러닝의 강력한 근거 중 하나입니다.

좀 더 구체적으로 설명하자면, 이 정리는 "하나 이상의 은닉층(hidden layer)을 가지는 feedforward neural network는 임의의 컴팩트(compact)한 범위에서 정의된 연속 함수를 임의의 정확도로 근사할 수 있다"는 것을 말합니다.

여기서 컴팩트란 입력값의 범위가 유한하다는 것을 의미합니다. 예를 들어, 0에서 1까지의 범위에서 정의된 연속 함수를 딥러닝 모델로 근사할 수 있다는 것을 보여줍니다.

하지만 이 정리가 의미하는 바는 딥러닝 모델이 항상 최적의 근사치를 제공한다는 것은 아닙니다. 딥러닝 모델은 과적합(overfitting)과 같은 문제로 인해 일반화 성능이 저하될 수 있습니다. 따라서 실제 모델을 구현할 때에는 이러한 문제들을 고려하여 모델을 구성해야 합니다.
