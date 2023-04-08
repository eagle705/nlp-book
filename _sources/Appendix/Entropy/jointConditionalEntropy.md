# Joint Entropy와 Conditional Entropy
## Joint Entropy (결합엔트로피)
지금까지 엔트로피를 공부할때 하나의 확률변수에 대해서 배웠었는데요. 이번에는 두 개 이상의 확률 변수에 따른 Entropy의 정의에 대해서 알아보려고 합니다.

Joint Entropy는 두개의 확률 변수 $X$, $Y$가 있고 그것의 Joint probability distribution이 있다고 할 때, 해당 확률 분포에 대해서 얻을 수 있는 엔트로피를 뜻합니다. 직관적으로는 X와 Y의 각각의 특정값이 동시에 나타나는 이벤트들로부터 얻을 수 있는 정보량의 기대값 혹은 평균이라고 생각 할 수 있습니다. 

$$
H(X, Y):=-\mathbb{E}_{x, y \sim p(X, Y)}[\log p(x, y)]
$$

- 이산 확률 변수
$
H(X, Y)=-\sum_i p_{X, Y}\left(x_i, y_i\right) \log p_{X, Y}\left(x_i, y_i\right)
$
- 연속 확률 변수
$
H(X, Y)=-\int_{-\infty}^{\infty} \int_{-\infty}^{\infty} p(x, y) \log p(x, y) d x d y
$

### Joint Entropy의 성질
Joint Entropy에는 몇가지 성질이 있는데요, 제일 먼저 $X=Y$ 일때의 경우에 대해서 알아보겠습니다.

#### $X=Y$ 인 경우
$$
\begin{aligned}
&\begin{array}{r}
p_{X, Y}\left(x_i, y_i\right)=p_X\left(x_i\right)=p_Y\left(y_i\right) \\
H(X)=H(Y)=H(X, Y)
\end{array}\\
\end{aligned}
$$
증명은 아래와 같습니다.
$$
\begin{aligned}
H(X, Y) & =-\sum_i p_{X, Y}\left(x_i, y_i\right) \log p_{X, Y}\left(x_i, y_i\right) \\
& =-\sum_i p_X\left(x_i\right) \log p_X\left(x_i\right)=H(X) \\
& =-\sum_i p_Y\left(y_i\right) \log p_Y\left(y_i\right)=H(Y)

\end{aligned}
$$
이에 대한 직관적인 의미는 다음과 같은데요.   
첫번째는, $X$와 $Y$가 같으므로 $X$를 알아냄을 통해서 얻을 수 있는 정보량은 $Y$를 알아냄을 통해서 얻을 수 있는 정보량과 같다. 즉 $H(X)=H(Y)$를 나타냅니다.   
두번째는, $X$만 아라앤도 $Y$를 자동으로 알 수 있으므로 $X$를 알아내어 얻는 정보량과 $X,Y$를 알아내어 얻는 정보량이 같다. 즉 $H(X)=H(X,Y)$를 나타냅니다.

#### $X, Y$가 통계적 독립인 경우
$X, Y$가 독립이라면, 확률에서는 다음과 같이 joint probability가 곱하기로 바뀌고 조건부 확률도 조건으로 주어진 변수가 확률 분포에 영향을 주지 않게 됩니다.  
엔트로피의 경우 joint probability가 곱하기로 바뀌게 되면서 log로 인해 엔트로피는 더하기 관계로 변환이 가능합니다.

$$
\begin{aligned}
& p_{X, Y}\left(x_i, y_i\right)=p_X\left(x_i\right) p_Y\left(y_i\right) \\
& p_X\left(x_i \mid y_i\right)=p_X\left(x_i\right) \\
& 
H(X) + H(Y) = H(X,Y)
\end{aligned}
$$

증명은 다음과 같습니다.  

$$
\begin{aligned}
H(X, Y) & =-\sum_i p_{X, Y}\left(x_i, y_i\right) \log p_{X, Y}\left(x_i, y_i\right) \\
& =-\sum_i p_X\left(x_i\right) p_Y\left(y_i\right) \log p_X\left(x_i\right) p_Y\left(y_i\right) \\
& =-\sum_i p_X\left(x_i\right) \log p_X\left(x_i\right)-\sum_i p_Y\left(y_i\right) \log p_Y\left(y_i\right) \\
& =H(X)+H(Y)
\end{aligned}
$$

2번째에서 3번째 라인으로 넘어가는 식은 두 변수가 독립이고 확률의 합은 1이 되므로 예를 들어, $-\sum_i p_X\left(x_i\right) p_Y\left(y_i\right) \log p_X\left(x_i\right)$를 $-\sum_i p_X\left(x_i\right) \log p_X\left(x_i\right)$로 표기할 수 있습니다.

즉 정리하면, $X,Y$가 통계적으로 독립일때 $H(X)+H(Y)=H(X,Y)$ 관계가 성립하고 $X$가 어떻게 나오던 $Y$와는 상관이 없고 $X, Y$를 모두 알아내서 얻는 정보는 $X$, $Y$가 어떻게 나오게 되는지 각각 알아내야함을 의미합니다.

예를들면, 주사위와 동전을 각각 던질때 동전이 앞면이 나왔다고해서 주사위가 어떤 값이 나올 수는 알 수 없을텐데요, 주사위와 동전 던지기는 서로 독립적인 행동이기 때문입니다. 

### 일반화
Joint Entropy는 일반적으로 다음의 수식을 만족시키는 것으로 알려져 있습니다.

$$
H(X), H(Y) \leq H(X, Y) \leq H(X)+H(Y)
$$

즉, $X,Y$를 "모두" 알아냈을때($H(X,Y)$) 정보량은 $X$ 또는 $Y$"만" 알아냈을때($H(X), H(Y)$) 정보량보다 크거나 같습니다. X와 Y가 같을 경우에만 같고, 같지 않을 경우에는 "모두" 알아내는 경우의 정보량이 더 큽니다.  
또한, $X,Y$를 "모두" 알아냈을때($H(X,Y)$) 정보량은 $X$와 $Y$를 각각 라아내어 얻어지는 정보량의 합보다는 작습니다. 독립이어야 같고, 독립이 아니면 각각의 합이 더 큽니다.  
즉, 고려해야할게 많을 수록 정보량은 더 커집니다. 변수가 같을때보다 같지 않을때 고려해야할게 더 많고, 독립적이지 않은 의존적인 관계일때보다 독립적일때 더 고려해야할 점이 많다고 기억하시면 좋을 것 같습니다.

## Conditional Entropy (조건부 엔트로피)
$X$를 이미 알고 있을 때 $Y$를 새로 알려주면 얻는 정보량의 기대값

$$
H(Y \mid X):=-\mathbb{E}_{x, y \sim p(X, Y)}[\log p(y \mid x)]
$$

- 이산 확률 변수
$$
H(Y \mid X)=-\sum_i p_{X, Y}\left(x_i, y_i\right) \log p_{X, Y}\left(y_i \mid x_i\right)
$$
- 연속 확률 변수
$$
H(Y \mid X)=-\int_{-\infty}^{\infty} p(x, y) \log p(y \mid x) d x
$$

### Conditional Entropy의 특성
$X, Y$를 모두 알기 위해 필요한 정보량($H(X,Y)$)은 $X$를 알기 위해 필요한 정보량($H(X)$)과 $X$가 주어졌을때 $Y$를 알기 위해 필요한 정보량($H(Y|X)$)의 합과 같다는 특성이 있습니다.

$$
H(X, Y)=H(Y \mid X)+H(X)=H(X \mid Y)+H(Y)
$$

증명은 다음과 같습니다. $x, y$는 joint probability($P(X,Y)$)를 따른다는 것을 기억하세요.

$$
\begin{aligned}
H(Y \mid X) & =-\sum_i p_{X, Y}\left(x_i, y_i\right) \log p_{X, Y}\left(y_i \mid x_i\right) \\
& =-\sum_i p_{X, Y}\left(x_i, y_i\right) \log \frac{p_{X, Y}\left(x_i, y_i\right)}{p_X\left(x_i\right)} \\
& =H(X, Y)-H(X)
\end{aligned}
$$
