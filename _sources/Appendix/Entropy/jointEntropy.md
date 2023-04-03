# Joint Entropy
지금까지 엔트로피를 공부할때 하나의 확률변수에 대해서 배웠었는데요. 이번에는 두 개 이상의 확률 변수에 따른 Entropy의 정의에 대해서 알아보려고 합니다.

Joint Entropy는 두개의 확률 변수 $X$, $Y$가 있고 그것의 Joint probability distribution이 있다고 할 때, 해당 확률 분포에 대해서 얻을 수 있는 엔트로피를 뜻합니다. 직관적으로는 X와 Y의 각각의 특정값이 동시에 나타나는 이벤트들로부터 얻을 수 있는 정보량의 기대값 혹은 평균이라고 생각 할 수 있습니다. 

$$
H(X, Y):=-\mathbb{E}_{x, y \sim p(X, Y)}[\log p(x, y)]
$$
- 이산 확률 변수
$$
H(X, Y)=-\sum_i p_{X, Y}\left(x_i, y_i\right) \log p_{X, Y}\left(x_i, y_i\right)
$$
- 연속 확률 변수
$$
H(X, Y)=-\int_{-\infty}^{\infty} \int_{-\infty}^{\infty} p(x, y) \log p(x, y) d x d y
$$
