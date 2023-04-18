# PEFT(Parameter Efficient Fine-Tuning)

안녕하세요, 오늘은 PEFT(Parameter Efficient Fine-Tuning)이라고 불리우는 기법에 대해서 알아보도록 하겠습니다. PEFT 기법중에서 가장 많이 알려진 것은 LoRA라는 기법인데요. 오늘은 LoRA에 대해서 간단하게 배우고 더 나아가 IA3라는 방법론에 대해서도 리뷰해보도록 하겠습니다.

## 라지 언어 모델의 발전과 PEFT의 필요성 대두
최근 GPT-3(175B)와 같은 매우 큰 언어모델이 등장함에 따라 언어모델에 해결하고자 하는 문제에 대한 몇가지 예시만 미리 입력해주면 in-context learning(ICL)에 의해 모델을 따로 튜닝할 필요 없이 문제를 쉽게 해결하는 것이 가능해졌습니다. 

[few-shot 평가 예시]

그러나 이러한 ICL은 매번 미리 예제를 입력해주어야하기 때문에 계산적인 부분이나 메모리, 저장비용들이 발생하게 된다는 단점이 있습니다. 또한 어떤 연구에서는 incorrect labels을 예제로 넣어주더라도 문제를 잘 해결하기도 하더라 라는 내용도 있어서 ICL의 결과를 온전히 신뢰하기 어렵다 라는 의견도 있습니다.   

오늘 다루게 되는 `PEFT(Parameter Efficient Fine-Tuning)`은 이러한 단점을 보완하는 대안적인 패러다임중 하나이고 적은양 파라미터(예를 들면 0.01%)를 학습함으로써 새로운 문제를 거의 비슷한 성능으로 풀 수 있게 하자는게 주된 목표입니다. PEFT에는 현재 다양한 방법론들이 연구되고 있고, 가장 유명한 방법론으로 LoRA가 있습니다. LoRA를 개선한 방법론들도 많이 나오고 있는 상황인만큼, LoRA에 대해서 간단하게 리뷰하고 IA3라는 더 개선된 방법에 대해서도 살펴보겠습니다.

## PEFT 기법들
초기에 PEFT을 위해 제안되었던 방법은 어댑터(`adapters`)를 사용하는 것입니다. 여기서 말하는 `adapater`란 기존에 이미 학습이 완료된 모델(pre-trained model)의 사이사이에 학습 가능한 작은 feed-forward networks를 삽입하는 구조를 말합니다. 이때 pre-trained model의 weights는 고정해놓고 학습 가능한 작은 feed-forward networks만 아키텍쳐 중간중간마다 추가함으로써 적은 수의 파라미터로 모델을 튜닝하는 기법입니다.   
이러한 어댑터기반의 방법론 외에도 `LoRA`, `prompt tuning`, `prefix tuning`등 다양한 방법론이 제안되었습니다.   
여러 방법론이 있지만 Stable diffusion이나 LLaMA, Alapaca에서도 많이 적용되는 Microsoft에서 공개한 `LoRA`라는 방법론이 현재로서는 제일 유명한 것 같습니다.


## LoRA
LoRA(**Lo**w-**R**ank **A**daptation)의 개념을 간단하게 설명하자면, 고정된 weights를 갖는 pretrained model에 학습이 가능한 rank decomposition 행렬을 삽입한것으로 중간중간 학습이 가능한 파라미터를 삽입했다는 점에서는 어댑터와 비슷하지만 구조적으로 조금 다르다고 할 수 있습니다.
적은 양의 파라미터로 모델을 튜닝하는 방법론이기 때문에 적은수의 GPU로 빠르게 튜닝할 수 있다는 장점이 있습니다. LoRA에서 나온 rank decomposition이라는 말이 처음에는 어렵게 느껴졌었는데요. 아래 보이는 그림에서 처럼 행렬의 차원을  `r` 만큼 줄이는 행렬과 다시 원래 크기로 키워주는 행렬의 곱으로 나타내는 것을 의미합니다.

![LoRA](https://user-images.githubusercontent.com/7252598/230259439-fe58295d-9879-41c8-9454-0ecbe27cacde.png)

위 그림처럼 레이어 중간중간마다 존재하는 hidden states `h`에 값을 더해줄수 있는 파라미터를 추가해줘서 모델의 출력 값을 원하는 타겟 레이블에 맞게 튜닝하는 것이 LoRA의 핵심 개념이라고 할 수 있습니다. 코드상으로는 아래와 같이 구현할 수 있는데요. 기존에 모델에서 사용하던 `Linear Layer`를 LoRA의 로직이 적용된 커스텀 클래스로 교체해주면 적용할 수 있습니다.
`if self.r > 0:` 라는 if 문이 추가된 부분이 LoRA가 적용된 부분입니다.

```python

class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False
    
    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
```

위 코드에서 주목할 부분은 `eval` 함수쪽인데요. 

```python
def eval(self):
    def T(w):
        return w.T if self.fan_in_fan_out else w
    nn.Linear.eval(self)
    if self.merge_weights and not self.merged:
        # Merge the weights and mark it
        if self.r > 0:
            self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling # 행렬을 합치는 부분
        self.merged = True
```

LoRA가 행렬 연산을 기반으로 하기 때문에 기존 행렬 $W$를 LoRA에서 사용하는 $A$, $B$ 행렬을 기반으로 $W'$로 다음과 같이 재구성할 수 있습니다.

$$W = W_0 + BA$$

이렇게 함으로써 얻을 수 있는 이점은 새롭게 학습한 파라미터를 기존에 학습된 pretrained model에 합쳐줌으로써 추가적인 연산이 필요하지 않게 되어 속도도 그대로 유지하면서 아키텍쳐의 변경도 필요없어지게 됩니다.

## IA3(Infused Adapter by Inhibiting and Amplifying Inner Activations)
IA3도 LoRA와 비슷한 방법으로 적은 파라미터만을 추가해서 모델을 튜닝할 수 있는 방법론 입니다. LoRA의 경우에는 hidden state에 새로운 값을 더해주는 기법이었다면 IA3의 경우에는 Attention에서 Key, Value 값을 rescale해주는 벡터와 position-wise feed-forward network의 값에 rescale을 해주는 벡터를 추가해서 모델을 튜닝해주는 기법입니다.

![IA3](https://user-images.githubusercontent.com/7252598/230011306-0e6184ac-af46-4920-b483-03e2481d0457.png)

IA3는 기존에 공개된 LoRA보다 적은 파라미터를 사용하면서 높은 성능을 내는 것으로 알려져있습니다.

![ia3_performance](https://user-images.githubusercontent.com/7252598/230011771-6be13446-6fd8-4a6d-a558-219cdd975eee.png)

IA3도 LoRA와 마찬가지로 Linear Layer를 커스텀 구현체로 변경함으로써 구현이 가능하며, LoRA Layer에 대한 configuration을 수정해서 구현할 수 있습니다. 다음은 IA3에 대한 구현체입니다.

- `/configs/ia3.json`
```json
{
    "lora_scaling_rank": 1,
    "lora_rank": 0,
    "lora_init_scale": 0.0,
    "lora_modules": ".*SelfAttention|.*EncDecAttention|.*DenseReluDense",
    "lora_layers": "k|v|wi_1.*",
    "trainable_param_names": ".*lora_b.*",
    "model_modifier": "lora",
    "lr": 3e-3,
    "num_steps": 1000
}
```

- lora기반 ia3 구현체

```python

def modify_with_lora(transformer, config):
    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(config.lora_modules, m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(config.lora_layers, c_name):
                    assert isinstance(
                        layer, nn.Linear
                    ), f"LoRA can only be applied to torch.nn.Linear, but {layer} is {type(layer)}."
                    setattr(
                        module,
                        c_name,
                        LoRALinear(layer, config.lora_rank, config.lora_scaling_rank, config.lora_init_scale),
                    )
    return transformer

class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank, scaling_rank, init_scale):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank
        self.scaling_rank = scaling_rank
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        if self.rank > 0:
            self.lora_a = nn.Parameter(torch.randn(rank, linear_layer.in_features) * init_scale)
            if init_scale < 0:
                self.lora_b = nn.Parameter(torch.randn(linear_layer.out_features, rank) * init_scale)
            else:
                self.lora_b = nn.Parameter(torch.zeros(linear_layer.out_features, rank))
        if self.scaling_rank:
            self.multi_lora_a = nn.Parameter(
                torch.ones(self.scaling_rank, linear_layer.in_features)
                + torch.randn(self.scaling_rank, linear_layer.in_features) * init_scale
            )
            if init_scale < 0:
                self.multi_lora_b = nn.Parameter(
                    torch.ones(linear_layer.out_features, self.scaling_rank)
                    + torch.randn(linear_layer.out_features, self.scaling_rank) * init_scale
                )
            else:
                self.multi_lora_b = nn.Parameter(torch.ones(linear_layer.out_features, self.scaling_rank))

    def forward(self, input):
        if self.scaling_rank == 1 and self.rank == 0:
            # parsimonious implementation for ia3 and lora scaling
            if self.multi_lora_a.requires_grad:
                # 이 부분에서 IA3의 곱하기가 일어나는듯!!
                hidden = F.linear((input * self.multi_lora_a.flatten()), self.weight, self.bias)
            else:
                hidden = F.linear(input, self.weight, self.bias)
            if self.multi_lora_b.requires_grad:
                hidden = hidden * self.multi_lora_b.flatten()
            return hidden
        else:
            # general implementation for lora (adding and scaling)
            weight = self.weight
            if self.scaling_rank:
                weight = weight * torch.matmul(self.multi_lora_b, self.multi_lora_a) / self.scaling_rank
            if self.rank:
                weight = weight + torch.matmul(self.lora_b, self.lora_a) / self.rank
            return F.linear(input, weight, self.bias)
```
