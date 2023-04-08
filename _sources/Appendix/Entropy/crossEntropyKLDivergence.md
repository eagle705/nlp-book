# Mutual Information, Cross Entropy 그리고 KL Divergence까지
Mutual Information $I(X, Y)$은 $X$와 $Y$에 대해서 공통으로 얻을 수 있는 정보량을 뜻합니다.   

Mutual Information $I(X, Y)$와 Joint Entorpy $H(X, Y)$, Conditional Entropy $H(X|Y)$의 관계는 다음과 같습니다.  
![mutual-information](https://user-images.githubusercontent.com/7252598/230711668-e5617283-21b7-4328-9532-b1cf6de6d35d.svg)

$$
\begin{aligned}
I(X, Y) & :=H(X, Y)-H(X \mid Y)-H(Y \mid X) \\
& =H(X)-H(X \mid Y) \\
& =H(Y)-H(Y \mid X) \\
& =H(X)+H(Y)-H(X, Y) \\
& =\mathbb{E}_{x, y \sim p(X, Y)}\left[\log \frac{p_{X, Y}(x, y)}{p_X(x) p_Y(y)}\right]
\end{aligned}
$$
