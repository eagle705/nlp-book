# 미분과 그래디언트

## 핵심 내용
- 단변수 함수의 미분
- 고차함수의 미분
- 선형 근사와 Taylor Series


### 단변수 함수의 미분 정의
- 함수가 어떤 지점에서 연속이고 극한값이 있을때, 해당 지점에서의 변화에 따른 함수 값의 변화량

![Tangent_animation](https://user-images.githubusercontent.com/7252598/228451197-d96bc84e-5c92-4089-85e5-2d09c45db37b.gif)

- 미분의 정의:

$$
\frac{d f}{d x}(x)=\lim _{\epsilon \rightarrow 0} \frac{f(x+\epsilon)-f(x)}{\epsilon}
$$

- 극한의 정의: 
$
\lim _{x \rightarrow c} f(x) \Leftrightarrow \lim _{x \rightarrow c+} f(x)=\lim _{x \rightarrow c-} f(x)
$
- 연속의 정의:
$
\lim _{x \rightarrow c} f(x)=f(c) \text { 이면 } x=c  { 에서 연속 }
$

![continuity](https://64.media.tumblr.com/6cdf62959b18340729980dca48deb1ce/332487a4ea1d26c6-2d/s1280x1920/904c3c12f2bb236cb96f74016b6c510fd4be30f1.gif)

### 다변수함수의 미분
현실 세계의 대다수의 문제는 다변수인경우(`이미지`, `단어벡터`, ...)가 많습니다. 이러한 다변수 문제를 다루기 위해 다변수 함수의 미분에 대해서 잠깐 살펴보겠습니다.
- 편미분의 정의:
  - 다변수 함수에 대해 "하나의 변수"를 제외하고 다른 변수들을 `상수`로 보고 함수의 단변수 미분을 하는 것

![image](https://user-images.githubusercontent.com/7252598/228452334-269c21eb-e488-4424-83c9-6e809bec550d.png)

저희가 풀고자하는 머신러닝, 딥러닝의 문제의 loss function은 다변수 함수로 모델링 할 수 있습니다. 일반적으로 Gradient의 방향을 따라서 이동하면 함수의 최대/최소점을 찾을 수 있게 됩니다. 그렇기 때문에 다변수함수 loss function을 최소화 하기 위해서 다변수 함수를 미분해서 `Gradient`를 구함으로써 loss를 최소화하는 방향으로 모델의 파라미터를 업데이트할 수 있습니다.
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
& \mathbf{x} \text { 에서 } \nabla_{\mathbf{x}} f(\mathbf{x}) \text { 방향으로 가면 } f(\mathbf{x}) \text { 가 가장 빠른 속도로 증가 } \\
& \text { 반대로 } \mathbf{x} \text { 에서 }-\nabla_{\mathbf{x}} f(\mathbf{x}) \text { 방향으로 가면 } f(\mathbf{x}) \text { 가 가장 빠른 속도로 감소 }
\end{aligned}
$$


Gradient Descent의 원리는 결국 Gradient의 반대방향으로 파라미터의 값을 조금씩 이동(learning rate) 시킴으로써 모델의 loss function의 값을 감소시키는 것입니다.

#### 다변수 함수에서의 Chain Rule
- 다변수 함수의 편미분에 대해서도 Chain rule이 기존과 동일하게 동작함
  - $f(u, v): f$ 는 $u, v$ 에 대한 함수
  - $u(x, y), v(x, y): u, v$ 는 $x, y$ 에 대한 함수일 때
- Chain Rule은 Back propagation을 할때 등장하는 개념

$$
\frac{\partial f}{\partial x}=\frac{\partial f}{\partial u} \frac{\partial u}{\partial x}+\frac{\partial f}{\partial v} \frac{\partial v}{\partial x}
$$

![image](https://user-images.githubusercontent.com/7252598/228459323-33ca01ce-986d-4b99-95f5-9d3d35ec068f.png)
