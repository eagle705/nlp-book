# 딥러닝 기초 수학 (미분)

## 핵심 내용
- 단변수 함수의 미분
- 고차함수의 미분
- 선형 근사와 Taylor Series


### 단변수 함수의 미분 정의
- 함수가 어떤 지점에서 연속이고 극한값이 있을때, 해당 지점에서의 변화에 따른 함수 값의 변화량
- 미분의 정의:

$$
\frac{d f}{d x}(x)=\lim _{\epsilon \rightarrow 0} \frac{f(x+\epsilon)-f(x)}{\epsilon}
$$

- 극한의 정의: 
$$
\lim _{x \rightarrow c} f(x) \Leftrightarrow \lim _{x \rightarrow c+} f(x)=\lim _{x \rightarrow c-} f(x)
$$
- 연속의 정의:
$$
\lim _{x \rightarrow c} f(x)=f(c) \text { 이면 } x=c  { 에서 연속 }
$$

### 다변수함수의 미분
- Gradient: 함수가 가장 빠르게 증가하는 방향, 기울기
$$
\begin{aligned}
& \frac{\partial f}{\partial x_1}>0 \text { 라면, } x_1 \text { 을 증가시키면 } f(\mathbf{x}) \text { 가 증가 } \\
& \frac{\partial f}{\partial x_1}<\frac{\partial f}{\partial x_2} \text { 라면, } x_1 \text { 보다 } x_2 \text { 를 증가시키면 } f(\mathbf{x}) \text { 가 더 빠르게 증가 }
\end{aligned}
$$

$$
\nabla_{\mathbf{x}} f(\mathbf{x}):=\left[\begin{array}{c}
\frac{\partial f}{\partial x_1} \\
\frac{\partial f}{\partial x_2} \\
\vdots \\
\frac{\partial f}{\partial x_d}
\end{array}\right]
$$

$$
\begin{aligned}
& f(\mathbf{x}) \text { : 다변수 함수의 벡터식 표기 }\left(\mathbf{x}=\left[x_1, x_2, \cdots, x_d\right]^T\right) \\
& \qquad \mathbf{x} \text { 에서 } \nabla_{\mathbf{x}} f(\mathbf{x}) \text { 방향으로 가면 } f(\mathbf{x}) \text { 가 가장 빠른 속도로 증가 }
\end{aligned}
$$
