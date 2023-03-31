[Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html)

## Hello, Alpaca?

최근 LLaMa이어서 아주 핫한 모델이 있습니다. 바로 Alpaca라는 모델인데요. 오늘은 Stanford에서 공개한 오픈소스인 Alpaca에 대해서 간단히 소개해보려합니다.
Meta에서 공개한 LLaMa라는 언어모델을 Stanford 박사과정 학생들이 사용자의 명령어에 언어모델이 잘 답변할 수 있도록 Instruction-following 데이터로 파인튜닝한 모델입니다.
언어모델은 기본적으로 다음 단어를 예측하는 문제를 풀기 때문에 일반적인 사용자의 명령어에 자연스럽게 답변하기가 어려운데요. 그럼에도 불구하고 ChatGPT 같은 모델이 답변을 잘하는 것은 사용자의 의도에 맞게 모델을 Instruction-following 데이터로 튜닝 (Alignment) 했기 때문이라고도 볼 수 있습니다. 결국 사용자가 언어모델을 잘 활용하기 위해서는 Instruction tuning은 꼭 거쳐야하는 관문이라고 할 수 있습니다.
LLaMa를 튜닝한 모델이니 아마 라마와 비슷한 생김새 가진 알파카라고 이름을 지은게 아닌가 싶네요🤔

![image.png](https://devocean.sk.com/editorImg/2023/3/23/e61a09ac00cbff86421a28127c228355177ca171f85f2757aee6d21df3c5fdbd)

Alpaca는 논문이 따로 발표되진 않았지만, 어떤 데이터로 어떻게 학습을 했는지 코드와 함께 공개가 되어있어서 현재시점에서도 LLaMa와 같이 많은 변형 및 어플리케이션이 나오고 있는데요. 지금부터 한번 알아보도록 하겠습니다.

## Alpaca를 왜 만들었을까?

Stanford 학생들은 ChatGPT, Claude, Bing Chat등 다양한 모델이 이미 훌륭한 성능을 보여주고 있지만 그럼에도 불구하고 아직은 부족한 점이 있다고 지적합니다. 예를 들면, 잘못된 정보를 생성하거나, 사회적인 편견 및 불편한 말들을 생성하는 것이죠. 이러한 문제를 해결하기 위해 학계와의 협업이 필요하지만 OpenAI의 `text-davinci-003`과 같은 모델은 접근하기 힘든 closed-source model이기 때문에 연구에 어려움이 있다고 말합니다🥲
마침 Meta에서 LLaMa를 공개했고, 기존에 알려진 연구를 바탕으로 훨씬 저렴한 비용으로 모델을 학습할 수 있도록, 데이터 및 모델 학습 방법을 재현 가능하도록 공개한 것으로 보입니다.
결과적으로, Alpaca는 text-davinci-003(175B)보다 훨씬 작은 7B 모델이지만 유사하게 동작한다고 합니다.
[Gradio 기반 데모 페이지](https://alpaca-ai.ngrok.io/)도 공개했는데, 접속은 가끔 안되는 것 같네요🤔
![image.png](https://devocean.sk.com/editorImg/2023/3/23/c0b4da8b95c774dfc242df286f8ddde97cb4aef73ed9ade7233f3d6d1d3fe741)
Alpaca는 academic research에 한해서만 사용이 가능하고 상업적 사용은 금지하고 있는데요.
이유는 LLaMa의 라이센스가 [non-commercial 라이센스](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform)라는점 그리고 [OpenAI의 tet-davinci-003에서 얻어낸 데이터를 활용했다는 점](https://openai.com/policies/terms-of-use)등을 이유로 제시하고 있습니다.

## 학습 방법

Alpaca는 기본적으로 7B 크기의 LLaMa를 Backbone으로 두고 Instruction tuning을 한 모델입니다.
모델은 이미 공개되어있기 때문에 가장 중요한 것은 데이터인데요. 기존에 Instruction-following 데이터를 생성하기 위한 많은 연구가 있었고 작년 12월인 최근에 공개된 [self-Instruct](https://github.com/yizhongw/self-instruct)라는 연구를 참고해서 데이터를 생성했습니다.
self-Instruct의 핵심은 LLM(Large Language Model)로 데이터를 생성해서 그 데이터로 다시 LLM을 학습한다는 것인데요, 한 마디로 튜닝을 위한 데이터도 모델이 생성하는 자가수급 시스템(?)이라고 볼 수 있습니다.
Alpaca에서는 self-Instruct의 방법론을 조금 단순화하되 모델은 더 좋은 모델([GPT-3(davinci) -> GPT-3.5(text-davinci-003)](https://platform.openai.com/docs/model-index-for-researchers/models-referred-to-as-gpt-3-5))을 사용해서 데이터를 생성했습니다.

##### 데이터 생성 예시

Alpaca에서 공개한 방법대로 데이터가 생성되는지 [OpenAI playground](https://platform.openai.com/playground)에서 테스트를 해보았습니다.
아래 보이는 예시와 같이 데이터 생성이 잘 되는 것을 확인 할 수 있습니다.

###### 한국어 결과
![alpaca-self-gen-korean](https://user-images.githubusercontent.com/7252598/225496927-2df4614a-8e35-4032-b65a-54e898ca61e6.gif)

###### 영어 결과
![alpaca-gen-self-instruct-en](https://user-images.githubusercontent.com/7252598/225496953-69f90efb-a20e-4147-bf4a-df623fc33e83.gif)

위와 같은 과정을 반복하면서 사람이 직접 만든 175개의 seed 데이터셋을 기반으로 데이터를 약 52,000개까지 추가 생산을 하고, 이 데이터를 학습셋으로 활용했습니다.
![image.png](https://devocean.sk.com/editorImg/2023/3/23/1ea18693744d12a835878563919dca568be6bc15017abc8541cf1e969aca5e0d)
데이터의 품질이 매우 뛰어나다고 할 수는 없겠지만 모델을 Alignment하기에는 어느정도 충분한 데이터를 생성한 것으로 보입니다. (52,000건의 데이터를 생성하는데는 `$500` 정도의 비용이 들었다고 합니다)
self-Instruct로 생성한 데이터로 A100(80GB) 8대로 Supervised Fintuning(SFT)하면 3 epoch에 3시간정도 소요되며, 일반적인 명령어에도 잘 답변할 수 있는 모델이 탄생하게 됩니다. (이때 발생한 비용은 $100 이하로 들었다고 합니다)


###### prompt
- requirement 템플릿에다가 seed_task 입력해놓음 default는 3개!
```python
def encode_prompt(prompt_instructions):
    """Encode multiple prompt instructions into a single string."""
    prompt = open("./prompt.txt").read() + "\n"

    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input = "<noinput>" if input.lower() == "" else input
        prompt += f"###\n"
        prompt += f"{idx + 1}. Instruction: {instruction}\n"
        prompt += f"{idx + 1}. Input:\n{input}\n"
        prompt += f"{idx + 1}. Output:\n{output}\n"
    prompt += f"###\n"
    prompt += f"{idx + 2}. Instruction:"
    return prompt

def generate_instruction_following_data(
    output_dir="./",
    seed_tasks_path="./seed_tasks.jsonl",
    num_instructions_to_generate=100,
    model_name="text-davinci-003",
    num_prompt_instructions=3,
    request_batch_size=5,
    temperature=1.0,
    top_p=1.0,
    num_cpus=16,
):
    seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
    seed_instruction_data = [
        {"instruction": t["instruction"], "input": t["instances"][0]["input"], "output": t["instances"][0]["output"]}
        for t in seed_tasks
    ]
    print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")

    os.makedirs(output_dir, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    machine_instruction_data = []
    if os.path.exists(os.path.join(output_dir, "regen.json")):
        machine_instruction_data = utils.jload(os.path.join(output_dir, "regen.json"))
        print(f"Loaded {len(machine_instruction_data)} machine-generated instructions")

    # similarities = {}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
    if machine_instruction_data:
        progress_bar.update(len(machine_instruction_data))

    # first we tokenize all the seed instructions and generated machine instructions
    all_instructions = [d["instruction"] for d in seed_instruction_data] + [
        d["instruction"] for d in machine_instruction_data
    ]
    all_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in all_instructions]

    while len(machine_instruction_data) < num_instructions_to_generate:
        request_idx += 1

        batch_inputs = []
        for _ in range(request_batch_size):
            # only sampling from the seed tasks
            prompt_instructions = random.sample(seed_instruction_data, num_prompt_instructions)
            prompt = encode_prompt(prompt_instructions)
            batch_inputs.append(prompt)
        decoding_args = utils.OpenAIDecodingArguments(
            temperature=temperature,
            n=1,
            max_tokens=3072,  # hard-code to maximize the length. the requests will be automatically adjusted
            top_p=top_p,
            stop=["\n20", "20.", "20."],
        )

```

````
20개의 다양한 task instruction 세트를 작성하라는 요청을 받습니다. 이러한 task instruction은 GPT 모델에 제공되며 instruction을 완료하기 위해 GPT 모델을 평가합니다.

요구 사항은 다음과 같습니다.
1. 다양성을 극대화하기 위해 각 instruction에 대해 동사를 반복하지 마십시오.
2. instruction에 사용되는 언어도 다양해야 합니다. 예를 들어, 명령형 instruction과 질문을 결합해야 합니다.
3. instruction의 종류가 다양해야 한다. 목록에는 open-ended 생성, 분류, 편집 등과 같은 다양한 유형의 작업이 포함되어야 합니다.
2. GPT 언어 모델은 instruction을 완료할 수 있어야 합니다. 예를 들어 어시스턴트에게 시각적, 사진, 이미지 또는 오디오 출력과 관련된 instruction을 생성하지 마십시요. 또 다른 예를 들면 어시스턴트에게 오후 5시에 깨우라고 요청하거나 어떤 작업도 수행할 수 없기 때문에 미리 알림을 설정하지 마십시오.
3. instruction는 한국어로 작성해야 합니다.
4. instruction은 1~2문장이어야 합니다. 명령형 문장이나 질문이 허용됩니다.
5. instruction에 대한 적절한 입력을 생성해야 합니다. 입력 필드에는 instruction에 대해 제공된 특정 예가 포함되어야 합니다. 현실적인 데이터를 포함해야 하며 단순한 자리 표시자를 포함해서는 안 됩니다. 입력 내용은 instruction을 어렵게 만들 수 있는 실질적인 내용을 제공해야 하지만 이상적으로는 100단어를 초과하지 않아야 합니다.
6. 모든 instruction에 입력이 필요한 것은 아닙니다. 예를 들어, instruction이 "세계에서 가장 높은 봉우리는 무엇입니까?"와 같은 일반적인 정보에 대해 묻는 경우 특정 컨텍스트를 제공할 필요가 없습니다. 이 경우 입력 필드에 "<noinput>"을 넣기만 하면 됩니다.
7. 출력은 instruction과 입력에 대한 적절한 응답이어야 합니다. 출력이 100단어 미만인지 확인하십시오.

20개 Task instruction 목록:
````


###### HF로 학습 및 FSDP 사용
- python 3.10 사용
```sh
torchrun --nproc_per_node=4 --master_port=<your_random_port> train.py \
    --model_name_or_path <your_path_to_hf_converted_llama_ckpt_and_tokenizer> \
    --data_path ./alpaca_data.json \
    --bf16 True \
    --output_dir <your_output_dir> \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LLaMADecoderLayer' \
    --tf32 True

```

# Assets released
- Demo: An [interactive demo](https://crfm.stanford.edu/alpaca/) for everyone to try out Alpaca.
- Data: [52K demonstrations](https://github.com/tatsu-lab/stanford_alpaca#data-release) used to fine-tune Alpaca.
- Data generation process: the code for [generating the data](https://github.com/tatsu-lab/stanford_alpaca#data-generation-process).
- Hyperparameters: for [fine-tuning](https://github.com/tatsu-lab/stanford_alpaca#fine-tuning) the model using the Hugging Face API.

## 공개된 Data 예시
![image](https://user-images.githubusercontent.com/7252598/225485522-f6c9a03e-2eac-4473-ba66-9d61b22217ff.png)

- 모델 및 학습 코드 공개
  - Training code: our code uses the [Hugging Face interface to LLaMA](https://github.com/huggingface/transformers/pull/21955). As of now, the effort to support LLaMA is still ongoing and not stable. We will give the exact training commands once Hugging Face supports LLaMA officially.
  - [아래에서 IGNORE_INDEX 부분이 중요!](https://github.com/tatsu-lab/stanford_alpaca/blob/61a3b4324505d284200a35dcbf1cc5e438ff2b46/train.py#L133)

```python
def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

```

## 마치며

Stable Diffusion과 같이 LLaMa도 공개가 되면서 다양한 후속 모델들이 빠르게 나오고 있습니다.
벌써 8-bit 양자화와 LoRA(Low-Rank Adaption)등을 활용해서 대형언어모델을 개인 PC에서도 쉽게 사용할 수 있게 하는 프로젝트들이 개발되고 있고, 이미 한국어로 튜닝된 [7B KoAlpaca 모델](https://github.com/Beomi/KoAlpaca), [65B 모델](https://huggingface.co/beomi/KoAlpaca-65B-LoRA?fbclid=IwAR3Damgr8gvwhqaSwwoOP-iHC5TepVhf9Gz2rlvpehOcAaplvGHjiH_RrBk)도 등장했습니다.
앞으로는 이전보다 더 저렴한 비용으로 더 괜찮은 성능의 모델들을 점점 더 많이 사용할 수 있게 될까요?
개선할 부분은 많겠지만 오픈소스 생태계에서 Alpaca의 등장은 그러한 미래에 분명 도움이 될 것으로 보입니다.
먼 미래가 아니라 지금이라도 조금은 더 똑똑하고 재미있는 자신만의 언어모델을 만들고 싶다면 지금 `Alpaca`를 사용해보시는건 어떨까요?🙂

## 참고자료

* [Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html)
* [Alpaca github](https://github.com/tatsu-lab/stanford_alpaca#authors)
* [KoAlpaca](https://github.com/Beomi/KoAlpaca)
