# MLE (Maximum Likelihood Estimation)
MLE는 베이지안 관점의 기계학습에서 유용하게 사용되는 개념입니다. 이제부터 MLE에 대해서 여러가지 예시와 함께 알아보도록 하겠습니다.

윷이 앞면이 나올 확률은 얼마일까요?
윷이 앞면이 나올 확률은 아마 거의 0.5겠지만 완전히 0.5는 아닐 것입니다. 여러가지 모양과 변수들이 있을 수 있으니까요.
정확한 확률을 알기 위해서는 직접 윷을 던져서 앞면이 나올 확률을 계산해 볼 수 있을 것입니다.

하나의 윷을 10회 던졌을때 결과가 다음과 같다고 해보겠습니다 (H: Head, 앞면, T: Tail 뒷면)

$$ HHTHTTHHHT $$

위 실험을 기반으로 계산한다면, 윷이 앞면이 나올 확률은 얼마일까요?

$$ {앞면이 나온 횟수\over전체 시행 횟수} = {6\over10} = 0.6 $$

이럴때 윷이 앞면이 나올 확률을 0.6이라고 할 수 있을까요?

조금 더 수식을 통해서 알아보도록 하겠습니다.

윷의 확률적 모델은 다음과 같습니다. 앞면이 나올 확률은 ${\theta}$로 가정하겠습니다.

$$
\begin{gathered}
P(x \mid \theta)=\left\{\begin{array}{cc}
\theta, & x=\mathrm{H} \\
1-\theta, & x=\mathrm{T}
\end{array}\right. \\
P(X \mid \theta)=P\left(x_1, x_2, \ldots, x_n \mid \theta\right)=P\left(x_1 \mid \theta\right) P\left(x_2 \mid \theta\right) \cdots P\left(x_n \mid \theta\right)
\end{gathered}
$$

여기서 만약 ${\theta}$를 알고 있다면 윷을 10번 던졌을때의 상황 ${X = HHTHTTHHHT}$가 발생할 확률을 계산할 수 있습니다. ${\theta}$를 알고 있기 때문에 윷을 던졌을때 나오는 결과에 해당하는 $X$의 분포도 그려볼 수 있을 것입니다. ($P(X \mid \theta)$)

그러나, 우리가 진짜로 궁금한 것은 ${X = HHTHTTHHHT}$가 나왔을 때의 ${\theta}$값 입니다($P(\theta \mid X)$). 다시 한번 말씀드리지만 우리는 윷이 나올 정확한 확률이 궁금한 것이기에 상황은 주어진것이지만 실제 윷의 확률은 모른다고 보고 있습니다.

상황($X$)이 주어진 상태에서 ${\theta}$를 구해야하기에 ${\theta}$는 일종의 분포를 갖게 됩니다. 상황($X$)을 일으킬 수 있는 ${\theta}$의 값은 다양할 수 있기 때문입니다. 
이때, 상황($X$)이 주어졌을때, 가장 확률 값이 높은 ${\theta}$를 찾는 방법은 뭘까요? ${\theta}$를 가장 적절하게 추정하기 위해 가장 확률 값이 높은 ${\theta}$를 찾으려 합니다.

## 베이지안 관점의 기계학습
베이지안 관점에서는 train 데이터 ${X}$로 모델 ${\theta}$를 학습하여 test 데이터 ${X'}$을 잘 맞추는 것을 목표로 합니다. 학습데이터로 모델을 학습하고 학습된 모델로 unseen 데이터를 예측하는 과정입니다. 이때 베이지안 관점에서는 ${X}$와 ${X'}$뿐만 아니라 모델을 학습해야하기 때문에 ${\theta}$까지도 확률 변수로 생각합니다.

즉, 학습데이터 ${X}$가 주어졌을때 이 ${X}$라는 상황을 가장 잘 나타낼 수 있는, 그런 상황이 가장 잘 발생할 수 있는 ${\hat{\theta}}$를 찾는 것이 베이지안 관점의 기계학습이다 라고 할 수 있겠습니다. 수식으로 나타내면 다음과 같습니다.

$$
\hat{\theta}=\underset{\theta}{\operatorname{argmax}} P(\theta \mid X)
$$

${X = HHTHTTHHHT}$가 나왔을때 윷의 움직임을 모델링할 가장 타당한 ${\theta}$를 찾는 것입니다.

하지만 여기서 문제가 발생합니다. ${P(\theta \mid X)}$와 같이 ${X}$가 주어졌을때 ${\theta}$를 예측하는 함수, 모델링 하는 함수는 찾기가 어렵습니다. 반대로 ${\theta}$가 주어졌을때 ${X}$를 모델링하는 ${P(X \mid \theta)}$를 위한 함수는 비교적 찾기가 쉽습니다. 이 문제에선 베르누이 분포를 쓰면 될 것 같습니다. 그러므로 이러한 상황을 해결하기 위해 우리는 베이즈정리(Bayes' Rule)를 사용할 수 있습니다. 베이즈 정리를 통해 ${P(X \mid \theta)}$로 ${P(\theta \mid X)}$를 간접적으로 모델링 할 수 있습니다.

$$
\begin{aligned}
\hat{\theta} & =\underset{\theta}{\operatorname{argmax}} P(\theta \mid X)=\underset{\theta}{\operatorname{argmax}} \frac{P(X \mid \theta) P(\theta)}{P(X)} \\
& =\underset{\theta}{\operatorname{argmax}} P(X \mid \theta){P(\theta)}
\end{aligned}
$$

$P(X)$ 는 $\theta$와 관련이 없기 때문에 고려하지 않아도 되기 때문에 다음 식에서 제외할 수 있습니다. 정리하면, $P(\theta \mid X)$의 최대값에 대한 $\theta$를 구하는 문제가 $P(X \mid \theta)$와 ${P(\theta)}$의 곱의 최대값에 대한 $\theta$를 구하는 문제로 변경됩니다. 

$P(X \mid \theta)$는 특정한 $\theta$값이 정해졌을때 $X$의 특정한 결과값이 나타내는 확률 값이라는 의미에서 `Likelihood`라고 부릅니다.
여기서 ${P(\theta)}$는 데이터와는 상관없는 모델이 가지는 ${\theta}$에 대한 고유의 확률 분포(`Prior distribution`)로 볼 수 있습니다. 
$P(\theta \mid X)$의 경우에는 "${\theta}$가 먼저 주어지고 그에 따라 ${X}$가 특정 값이 나올 확률이 정해지는 $P(X \mid \theta)$와 같은 자연 스러운순서에 반대되는 순서"를 가지고 있기 때문에 ${\theta}$에 대한 `Posterior distribution`이라고 부릅니다.

## Likelihood의 정의

$$
\text { Likelihood }:=P(X \mid \theta)
$$

Likelihood는 ${\theta}$를 특정한 값으로 가정했을때, 주어진 학습 데이터 ${X}$가 얼마나 높은 확률 값을 가지는가에 대한 정보로 생각해 볼 수 있습니다. 즉, ${\theta}$가 적절하게 셋팅이 되었다면, 주어진 데이터 ${X}$가 발생할 가능성도 높아지므로 $P(X \mid \theta)$도 큰 값이 나올 것이다 라고 예상해 볼 수 있을 것입니다. 반면 ${\theta}$에 이상한 값이 셋팅되었다면 $P(X \mid \theta)$의 값은 매우 낮게 나올 것입니다.
그러므로 우리는 기존에 구하려했던 $X$가 주어졌을때 ${\theta}$의 확률을 구하려 했던 posterior distribution $P(\theta \mid X)$를 생각하기 이전에 ${ Likelihood }:=P(X \mid \theta)$를 Objective function으로 사용해서 Likelihood를 가장 큰 확률 값으로 만들어주는 최적의 ${\theta}$ 값을 찾는 문제를 생각할 수 있습니다.
이를 우리는 `Maximum Likelihood Estimation(MLE)`라고 부릅니다.

$$
\hat{\theta}_{M L E}=\underset{\theta}{\operatorname{argmax}} P(X \mid \theta)
$$

MLE는 다음과 같은 상황에서 쓰일 수 있습니다. posterior를 최대화하는 ${\theta}$를 찾는 문제는 결국 베이즈정리에 의해 아래와 같이 $P(X \mid \theta){P(\theta)}$에 대한 문제로 변경이 가능한데요.

$$
\begin{aligned}
\hat{\theta} & =\underset{\theta}{\operatorname{argmax}} P(\theta \mid X)=\underset{\theta}{\operatorname{argmax}} \frac{P(X \mid \theta) P(\theta)}{P(X)} \\
& =\underset{\theta}{\operatorname{argmax}} P(X \mid \theta){P(\theta)}
\end{aligned}
$$

Prior인 ${P(\theta)}$에 대해서 어떠한 값도 가질 수 있다는 관점(가설)을 갖고 문제를 푼다면 ${P(\theta)}$는 uniform distribution이 되고 상수$C$ 로 가정해서 $argmax$에서는 고려하지 않고 문제를 풀 수 있게 됩니다. 이러한 가정을 사용하면 식은 다음과 같이 변경되며 MLE 문제로 보고 풀 수 있게 됩니다.

$$
\begin{aligned}
\hat{\theta} & =\underset{\theta}{\operatorname{argmax}} P(\theta \mid X)=\underset{\theta}{\operatorname{argmax}} \frac{P(X \mid \theta) P(\theta)}{P(X)} \\
& =\underset{\theta}{\operatorname{argmax}} P(X \mid \theta){P(\theta)} =\underset{\theta}{\operatorname{argmax}} P(X \mid \theta)
\end{aligned}
$$

그렇다면 MLE 관점에서 문제를 풀어보겠습니다. 

$$
P(x \mid \theta)=\left\{\begin{array}{cc}
\theta, & x=\mathrm{H} \\
1-\theta, & x=\mathrm{T}
\end{array}\right.
$$

$$
\begin{aligned}
\hat{\theta}_{M L E} & =\underset{\theta}{\operatorname{argmax}} P(X \mid \theta)=\underset{\theta}{\operatorname{argmax}} \prod_{i=1}^{10} P\left(x_i \mid \theta\right) \\
& =\underset{\theta}{\operatorname{argmax}} \theta \theta(1-\theta) \theta(1-\theta)(1-\theta) \theta \theta \theta(1-\theta) \\
& =\underset{\theta}{\operatorname{argmax}} \theta^6(1-\theta)^4
\end{aligned}
$$

위 값을 미분했을때 0이 되는 지점이 극대 혹은 극소 값이 될 것입니다. 미분해보면 다음과 같습니다.

$$
\begin{aligned}
\frac{d}{d \theta} \theta^6(1-\theta)^4 & =6 \theta^5(1-\theta)^4-4 \theta^6(1-\theta)^3 \\
& =\theta^5(1-\theta)^3(6(1-\theta)-4 \theta) \\
& =\theta^5(1-\theta)^3(6-10 \theta)
\end{aligned}
$$

위 식을 풀어보면 ${\theta} = 0, 1$  인 지점이 극소값, ${\theta} = 0.6$ 인 지점이 극대 값으로 $\hat{\theta}_{M L E}=0.6$임을 알 수 있습니다.


## Log-likelihood
데이터에 대한 Likelihood를 나타내려면 데이터 수 만큼 확률 값을 곱해야합니다

$$
P(X \mid \theta)=P\left(x_1 \mid \theta\right) P\left(x_2 \mid \theta\right) \cdots P\left(x_N \mid \theta\right)
$$

$P\left(x_i \mid \theta\right) \leq 1$인 경우 반복적으로 곱하면 ${P(X \mid \theta)}$가 기하급수적으로 0에 가까워지게 됩니다. 0에 근접하게 되는 경우 컴퓨터상에서 연산이 정확하지 않을 수 있기 때문에 Log를 통해 다음과 같이 likelihood의 값을 변환해주게 됩니다.

$$
\log P(X \mid \theta)=\log P\left(x_1 \mid \theta\right) P\left(x_2 \mid \theta\right) \cdots P\left(x_N \mid \theta\right)=\log P\left(x_1 \mid \theta\right)+\cdots+\log P\left(x_N \mid \theta\right)
$$

아래 보이는 그림과 같이 Log scale로 바뀐 경우 0으로 값이 모이는 문제가 해결된걸 확인 할 수 있습니다

![image](https://user-images.githubusercontent.com/7252598/228896315-25ff202c-f22e-4126-8a0c-f21ea92371f4.png)
