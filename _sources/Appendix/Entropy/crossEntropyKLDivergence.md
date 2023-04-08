# Mutual Information, Cross Entropy 그리고 KL Divergence까지
Mutual Information $I(X, Y)$은 $X$와 $Y$에 대해서 공통으로 얻을 수 있는 정보량을 뜻합니다.   
예를 들어, 오늘 비가 왔는지에 대한 정보는 내일 비가오는지에 대한 정보 $X$와 모레 비가 오는지에 대한 정보 $Y$, 모두를 알아내는데 도움이 되는 Mutual Information이라고 할 수 있습니다.

Mutual Information $I(X, Y)$와 Joint Entorpy $H(X, Y)$, Conditional Entropy $H(X|Y)$의 관계는 다음과 같습니다.  
![image](https://user-images.githubusercontent.com/7252598/230729182-6f231b4c-b87b-4cf5-a527-c92aab6b63e6.png)

$$
\begin{aligned}
I(X, Y) & :=H(X, Y)-H(X \mid Y)-H(Y \mid X) \\
& =H(X)-H(X \mid Y) \\
& =H(Y)-H(Y \mid X) \\
& =H(X)+H(Y)-H(X, Y) \\
& =\mathbb{E}_{x, y \sim p(X, Y)}\left[\log \frac{p_{X, Y}(x, y)}{p_X(x) p_Y(y)}\right]
\end{aligned}
$$

$X, Y$에 대한 정보를 모두 모은 후에 $X$에서 $Y$와 관계된 정보가 빠진 정보량과 반대로 $Y$에서 $X$와 관계된 정보가 빠진 정보량을 빼는 것으로 구할 수 있습니다. 여기서 기호 $I$는 Mutual information을 의미합니다.

$$
I(X, Y):=H(X, Y)-H(X \mid Y)-H(Y \mid X)
$$

Mutual information은 Conditional Entropy를 통해서도 계산할 수 있는데요. $X, Y$가 연관되어 있는 정보들을 모두 모으는 또 다른 방법으로 $Y$의 총 정보량에서 $X$를 알려줬을때의 $Y$에 대한 정보량을 빼는 것으로 구할 수 있습니다.

$$
\begin{aligned}
I(X, Y) & =H(Y)-H(Y \mid X) \\
& =H(X)-H(X \mid Y)
\end{aligned}
$$

# Cross Entropy (크로스 엔트로피)

$$
\mathrm{CE}(P, Q)=-\mathbb{E}_{x \sim p(x)}[\log q(x)]
$$

여기서 $P, Q$는 각각 $p(x), q(x)$를 따르는 확률 변수입니다.  
크로스 엔트로피는 엔트로피의 두번째 특성에서 언급했던 $-\mathbb{E}_{x \sim p(x)}[\log q(x)]$를 활용합니다.   
두번째 특성은 다음과 같았습니다.

$$
H(X)=\mathbb{E}_{x \sim p(x)}[-\log p(x)] \leq \mathbb{E}_{x \sim p(x)}[-\log q(x)]
$$

크로스 엔트로피는 $p(x)$에서 뽑히는 $x$를 가지고 $q(x)$로 놀라고 있는 것이라고 이해할 수 있는데요. $-\mathbb{E}_{x \sim p(x)}[\log q(x)]$의 하한은 $H(P)$이며 등호는 $p(x)=q(x)$일 때 성립됩니다. 
이러한 상황에서 $q(x)$를 모델링 할때, $CE(P, Q)$를 최소화하면 $q(x)$를 $p(x)$와 `최대한 같도록` 모델링 할 수 있게됩니다! 이러한 원리를 활용해서 머신러닝 방법론에서 Cross entropy loss를 많이 사용하게 되었습니다.


# Kullback-Leibler Divergence (KL divergence)
KL divergence는 한 분포에서 다른 분포까지 떨어진 확률 분포간의 거리라고 말하곤 하는데요. 대칭적이지 않기 때문에($D_{K L}(P \| Q) \neq D_{K L}(Q \| P)$) 엄밀하게 말하면 거리는 아닙니다만, 사용하기에 좋은 여러가지 특징들이 존재합니다.


$$
D_{K L}(P \| Q):=\mathbb{E}_{x \sim p(x)}\left[\log \frac{p(x)}{q(x)}\right]
$$

크로스 엔트로피와의 관계는 다음과 같습니다.

$$
D_{K L}(P \| Q)=\operatorname{CE}(P, Q)-H(P)
$$

그러므로 $q(x)$를 모델링 할 때 크로스 엔트로피를 최소화하는 것은 KL divergence를 최소화 하는 것과 동일하게 동작한다고 할 수 있습니다.

$$
\underset{q(x)}{\operatorname{argmin}} \operatorname{CE}(P, Q)=\underset{q(x)}{\operatorname{argmin}}\left(D_{K L}(P \| Q)+H(P)\right)=\underset{q(x)}{\operatorname{argmin}} D_{K L}(P \| Q)
$$

KL divergence의 특징중 하나는 위에서 얘기했던것 처럼 대칭적이지 않기 때문에($D_{K L}(P \| Q) \neq D_{K L}(Q \| P)$) 교환법칙이 성립하지 않습니다. 또 하나의 특징으로는 모든 $P, Q$에 대해서 $D_{K L}(P \| Q) \geq 0$ 이며 등호성립조건은 $P = Q$입니다. 이러한 특징은 KL divergence가 항상 음이 아닌 실수라는 의미이고, 두 분포간의 거리 개념으로 쓰일 수 있게 합니다. 등호성립조건을 활용해서 $D_{K L}(P \| Q) > 0$인 경우 $P \neq Q$라는 것도 유추해낼 수 있습니다. 증명은 다음과 같습니다.   

1. Entropy의 특성 2: $H(P)$ 는 $-\mathbb{E}_{x \sim p(x)}[\log q(x)]$ (Cross-entropy)의 하한이다
2. $\mathrm{KL}$ divergence와 Cross-entropy와의 관계: $D_{K L}(P \| Q)=\operatorname{CE}(P, Q)-H(P)$
$$
\therefore D_{K L}(P \| Q)=\operatorname{CE}(P, Q)-H(P)=-\mathbb{E}_{x \sim p(x)}[\log q(x)]-H(P) \geq 0
$$


마지막으로 또 하나의 특징은 $p(x) > 0$이고 $q(x) = 0$인 지점이 존재할 경우 $D_{K L}(P \| Q)=\infty$라고 할 수 있습니다. $q(x)=0$인 지점에서 $x$가 뽑히면 무한히 놀랍다 라고 해석할 수 있습니다. $D_{K L}(P \| Q)$을 줄이는 경우에는 보통 $p(x)>0$이고 $q(x)=0$인 지점부터 없애려고 합니다.
