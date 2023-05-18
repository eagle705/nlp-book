# Meta가 쏘아올린 작은공 LLaMA

![llama](https://user-images.githubusercontent.com/7252598/231923020-9b1dea6d-8504-410e-a733-ef550b6c2591.png)

이번 챕터에서는 Meta에서 2023년에 공개한 언어모델인 LLaMA에 대해서 알아보고자 합니다.
LLaMA가 공개되면서 오픈소스 진영에서도 다양한 모델들 예를 들면 Alpaca와 같은 변형들이 등장하게 되었는데요. LLM 학습에 대한 스탠다드 레시피를 제공한 LLaMA 모델은 어떤 특징이 있는지 알아보겠습니다.

첫번째 특징은 학습에 사용한 Corpus입니다. OpenAI의 GPT-3나 Google의 PaLM과 같은 모델은 공개된 코퍼스외에 여러가지 사내에서 구축된 코퍼스를 사용하였는데요. LLaMA에서는 이것과 달리 모두 공개된 Copus만을 학습에 사용했습니다. 공개된 코퍼스만을 사용해도 좋은 품질의 LLM을 만드는 방법을 공개했다는 점에서 큰 의의가 있다고 할 수 있습니다.

두번째 특징은 학습에 사용한 Corpus의 양입니다. 언어모델을 학습할때 필요한 자원과 언어모델의 크기간의 관계에 대해서 연구한 딥마인드의 Chinchilla 논문에서는 모델의 크기가 10B정도의 수준일때 205.1B개의 토큰이 필요하다고 주장했었는데요. 
<img width="480" alt="image" src="https://github.com/eagle705/nlp-book/assets/7252598/aaaf5425-b0b8-4e8d-9463-54bc65946852">
LLaMA에서는 그보다 많이 학습해도 성능이 계속 올라간다는것을 알아냈습니다. 총 1T 토큰을 학습함으로써 모델 크기에 따라 어느정도의 토큰이 필요한지 알 수 있게 해주었습니다. LLaMA를 재현하는 프로젝트인 RedPajama에서도 이와 같은 조건으로 학습을 진행했었는데요. 
<img width="785" alt="image" src="https://github.com/eagle705/nlp-book/assets/7252598/f3422adb-6991-47e1-b7bc-fcdc4f5e270c">

같은 7B 모델이라도 학습하는 토큰의 개수가 많아질수록 더 성능이 올라가는 모습을 보여주었습니다. 참고로 그래프를 보시면 같은 크기를 갖는 다른 모델들과 비교해도 LLaMA의 성능이 꽤 높은 좋은 모델임을 확인할 수 있습니다.

마지막 특징은 기존에 공개된 Transformer 구조를 수정 및 개선해서 모델링했다는 점입니다.

## LLaMA Architecture
LLaMA 모델에서는 아래와 같이 크게 3가지 부분에서 변화를 주었습니다.

### Pre-normalization
맨처음 Transformer가 발표되었을때는 Multi-Head Attention과 Residual 연산 이후에 Normalization을 사용했었습니다. 이를 Post-LN이라고 부릅니다. 하지만 후속연구가 진행되면서 GPT-3부터는 Multi-Head Attention에 입력으로 들어가기전에 미리 Normalization을 해주는 것이 학습을 더 안정시킨다고 알려지게 되었습니다. 이를 Pre-LN이라고 부르며 이후 모델들은 이 방법을 주로 사용하게 되었습니다. Normalization은 T5에서 주로 사용되었던 RMSNorm을 도입하게 되었습니다.

<img width="311" alt="image" src="https://github.com/eagle705/nlp-book/assets/7252598/017bed00-0618-47b4-a8d5-70e70666715f">

### SwiGLU
이전의 대부분의 모델은 Activation Function을 GELU, ReLU등을 사용했었습니다.하지만 구글에서 만든 PaLM이 발표되면서 LLaMA 모델에서는 SwiGLU가 도입되었습니다.


### Rotary Embedding
GPT에서는 Absolute, learnable, Rotary, ALiBi등 다양한 Positional Embedding이 사용되고 있었습니다. 하지만 최근에 나온 여러 연구에서는 주로 Rotary Embedding을 쓰고 있습니다. Rotary Embedding은 오픈소스 진영인 EleutherAI에서 만든 GPTNeo에서 제안된 방법론이며 Embedding의 각도를 이동시켜 위치정보를 주입하는 방법을 통해 적용됩니다.

<img width="198" alt="image" src="https://github.com/eagle705/nlp-book/assets/7252598/a129b308-3f8b-451a-9e90-f2cf0ddbf7f8">

## 마치며
LLaMA는 Meta에서 연구목적으로 공개한 SOTA 성능을 가진 LLM입니다. 공개된 Corpus로만 학습을 했고 같은 크기의 Chinchilla나 PaLM, GPT-3에 비해서도 좋은성능을 내는 결과를 보여주었습니다. 허깅페이스를 통해서는 다음과 같이 사용할 수 있습니다.

```python
from peft import PeftModel
from transformers import LLaMATokenizer, LLaMAForCausalLM, GenerationConfig

model_name = "decapoda-research/llama-7b-hf"
tokenizer = LLaMATokenizer.from_pretrained(model_name)

# 🤗 Transformers is closely integrated with most used modules on bitsandbytes. You can load your model in 8-bit precision with few lines of code.
model = LLaMAForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto",
)
```
