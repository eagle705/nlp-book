# Transformer models
## Introduction
본 챕터에서는 [Transformers](https://github.com/huggingface/transformers), [Datasets](https://github.com/huggingface/datasets), [Tokenizer](https://github.com/huggingface/tokenizers), [Accelerate](https://github.com/huggingface/accelerate)와 같은 🤗 허깅페이스 에코시스템을 통해 자연어처리를 하는 방법에 대해 다룰 예정입니다.

## 배우게 될 내용들

앞으로 배우게 될 내용은 크게 3단계로 나누어서 진행될 예정입니다.

1단계에서는 `Transformers` 라이브러리의 메인 컨셉에 대해 학습합니다. transformers 라이브러리가 어떻게 동작하는지, [HugggingFace Hub](https://huggingface.co/models)에 공유된 사전 학습 모델을 어떻게 사용하는지, 모델은 어떻게 파인튜닝하고, 학습한 모델은 어떻게 Hub에 공유하는지에 대해 학습할 예정입니다.  

2단계에서는 본격적인 NLP task를 하기 앞서 `Datasets`과 `Tokenizer` 사용법에 대해 배울 예정입니다.

3단계에서는 스페셜한 아키텍쳐들 (memory efficiency, long sequences, etc.) 및 학습속도 향상등 다양한 테크닉에 대해 다룰 예정입니다.

## 사전지식
- 파이썬에 익숙한 사람
- 1개 이상의 딥러닝 수업을 수강한 사람 (PyTorch 또는 TensorFlow에 대한 선행지식까지는 필요없음)
