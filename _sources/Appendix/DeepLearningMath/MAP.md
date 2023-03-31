# MAP (Maximum A P Posteriori Estimation)
이전 챕터에서 다뤘던 MLE의 조금 변형된 버전인 MAP에 대해서 배워보도록 하겠습니다. 

지난번에 배웠던 윷 던지기 예제에서는 아래와 같이 argmax 문제를 푸는 경우

$$
\begin{aligned}
\hat{\theta} & =\underset{\theta}{\operatorname{argmax}} P(\theta \mid X)=\underset{\theta}{\operatorname{argmax}} \frac{P(X \mid \theta) P(\theta)}{P(X)} \\
& =\underset{\theta}{\operatorname{argmax}} P(X \mid \theta){P(\theta)} 
\end{aligned}
$$

Posterior $P(\theta \mid X)$를 $P(X \mid \theta){P(\theta)}$로 보고 문제를 풀 수 있고 여기서 Prior인 $P(\theta)$ 값을 `상수로 가정`(어떤 $\theta$도 가질 수 있다고 생각하는 관점)하고 argmax에 영향을 주지 않는다고 보고 Likelihood $P(X \mid \theta)$에 대해만 고려해서 Likelihood를 최대로 하는 $\hat{\theta}$를 찾는 방식으로 문제를 MLE  방법론으로 치환해서 풀었었습니다.

위에서는 Prior인 $P(\theta)$를 상수로 가정했지만, 실제로도 그럴까요? 아마 실제로는 그렇지 않을 겁니다. 분명 어떤 분포를 갖거나 특징이 있겠죠. 즉 MLE는 상수로 가정한 스페셜 케이스에 대한 풀이라고 볼 수 있고 $P(\theta)$에 대한 부분도 조금 더 고려한다면 좀 더 정확한 방법으로 문제를 해결 할 수 있을 것 입니다.

저희는 윷이 앞면이 나올 확률이 대략 0.5에 근접하다는 사실을 사전에 미리 알고 있습니다. 즉 직관적으로 $P(\theta = 0.5) > P(\theta = 0.001)$ 라는걸 알고 있는 거죠. 이미 여기서부터 $P(\theta)$는 사실 uniform distribution이라고 할 수 없고, 즉 상수라고 볼 수는 없는겁니다. 하지만 이 $P(\theta)$값을 정확히 알 방법은 없습니다. 예컨데 윷의 모양을 본다면? 앞면이 나올 확률이 딱 0.5라고 할 수 없는 것이죠. 그래도 $P(\theta)$를 적당히 가정한다면 $\hat{\theta}$ 값을 구하는 과정을 보완할 수 있지 않을까요?

### 용어정리
여기서 잠깐 용어를 정리하고 가겠습니다.
- Posterior distribution (사후 분포) $P(\theta \mid X)$: 데이터를 관측한 후의 $\theta$의 분포 (데이터를 본 후 정보를 반영한 ${\theta}$의 분포)
- Prior distribution (사전 분포) $P(\theta)$: 데이터를 관측하기 전의 $\theta$의 분포
- Maximum A Posterior Estimation (MAP): Likelihood에 Prior를 곱해서 Posterior를 최대화 하는 것 (하지만 Prior를 정확히 알수 없기 때문에 `가정`해서 곱해줄 수 밖에 없음)

### MAP 계산을 위한 윷놀이 예시
자 그럼 이번에도 윷던지기 예시를 통해서 MAP를 계산하는 과정을 한번 살펴보고자 합니다.  

하나의 윷을 10번 던졌을 때, 다음과 같은 결과가 나왔다고 해보겠습니다.

$HHTHTTHHHT$ ((H: Head 앞면, T: Tail 뒷면)

윷을 수학적으로 모델링하면 다음과 같이 나타낼 수 있을것 입니다.

$$
\begin{gathered}
P(x \mid \theta)=\left\{\begin{array}{cc}
\theta, & x=\mathrm{H} \\
1-\theta, & x=\mathrm{T}
\end{array}\right. \\
P(X \mid \theta)=P\left(x_1, x_2, \ldots, x_n \mid \theta\right)=P\left(x_1 \mid \theta\right) \times P\left(x_2 \mid \theta\right) \times \cdots \times P\left(x_n \mid \theta\right) 
\end{gathered}
$$

윷의 경우 앞면이 나올 확률이 아마 0.5 주변에 있을 가능성이 크기 때문에 0.5일 확률이 가장 높은 임의의 분포 만들어서 `가정`해보면 다음과 같은 식을 만들 수 있을 것입니다. 이때 가정한 분포 또한 확률이므로 분포의 Probability density function의 넓이는 1을 만족해야 합니다.

$$
\begin{gathered}
\text { 사전 분포 } P(\theta)=-4|\theta-0.5|+2
\end{gathered}
$$

![image](https://user-images.githubusercontent.com/7252598/229012700-59bc5fa9-a805-4e7a-a5ba-4227906193e8.png)

MAP 관점에서 최적의 $\theta$는 어떻게 될까요?

$$
\begin{aligned}
\hat{\theta}_{M A P} & =\underset{\theta}{\operatorname{argmax}} P(X \mid \theta) P(\theta) \\
& =\underset{\theta}{\operatorname{argmax}} \theta^6(1-\theta)^4(-4|\theta-0.5|+2)
\end{aligned}
$$

위와 같이 표현할 수 있을 것입니다. 위 식을 풀기 위해 미분을 하면 다음과 같이 정리할 수 있습니다.

$$
\begin{array}{ll}
\frac{d}{d \theta} \theta^6(1-\theta)^4(-4 \theta)=4 \theta^6(11 \theta-7)(1-\theta)^3=0 & \theta \in[0,0.5] \\
\frac{d}{d \theta} \theta^6(1-\theta)^4(-4 \theta+4)=-4 \theta^5(11-6 \theta)(1-\theta)^4=0 & \theta \in[0.5,1]
\end{array}
$$

첫번째 범위에 해당하는 미분식에서 $\theta=0$, 1$인 지점은 극소값에 해당하고 $\theta=\frac{7}{11}$ 은 범위 밖에 해당하므로 제외할 수 있습니다. 

두번째 범위에 해당하는 미분식에서는 $\theta=\frac{6}{11}$ 인 지점이 극대값에 해당하므로 다음과 같이 $\hat{\theta}$를 구하기 위한 MAP를 계산하면 아래와 같은 결과를 얻을 수 있습니다.

$$
\therefore \hat{\theta}_{M A P}=\frac{6}{11}
$$

그래프로 확인해보면 아래와 같이 Likelihood의 분포가 특정 분포로 가정된 Prior 분포와 곱해지면서 아래와 같이 보정된 Posterior 분포를 얻을 수 있습니다.

![image](https://user-images.githubusercontent.com/7252598/229033660-f38d9b19-2744-4d84-a448-a964772af4bf.png)

### MAP의 장단점
MAP의 장점은 결국 적절한 사전분포를 모델링 할 수 있다면 Overfitting을 방지할 수 있다는 것입니다. 만약 윷놀이 데이터가 정말 우연치않게 3번 던졌을때 모두 앞면이 나온다면 MLE는 값이 1이 나오지만 MAP는 앞에서 사용했던 Prior를 사용한다면 0.75라는 조금 더 0.5에 가까운 값을 얻어낼 수 있게 됩니다.

$$\hat{\theta}_{M L E}=1,  \hat{\theta}_{M A P}=0.75$$

반면 MAP의 단점은 적절하지 않은 사전분포를 사용할 경우 모델링에 방해가 될 수 있습니다. 사실 윷은 모양이 볼록하기 때문에 앞면이 나올 확률이 0.5보다 작을 가능성이 높기 때문입니다. 이런 것들을 고려해서 Prior를 셋팅해야하기 때문에 MAP도 장점과 단점 모두 가지고 있다고 할 수 있습니다.
