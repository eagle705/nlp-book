# Introduction

Transformer는 자연어 처리 분야에서 가장 혁신적인 딥러닝 아키텍처 중 하나로, 2017년에 발표된 논문 `"Attention is All You Need"`에서 처음 소개되었습니다. 기존의 순환 신경망(RNN)과 달리, Transformer는 self-attention 메커니즘을 사용하여 입력 시퀀스 내의 모든 위치들 간의 상호작용을 쉽게 학습할 수 있습니다.

Transformer는 기계 번역, 질문 답변, 요약 등 다양한 자연어 처리 태스크에서 뛰어난 성능을 보이며, 이는 이전까지 RNN이 가지고 있던 장점과 함께, 더 빠른 학습과 높은 병렬성 등의 이점을 제공합니다. 또한, Transformer는 GPT, BERT 등의 대표적인 언어 모델 구조의 기반으로 사용되어, 자연어 처리 분야에서 큰 영향력을 발휘하고 있습니다.

Transformer는 인코더와 디코더로 구성되며, 인코더는 입력 시퀀스를 인코딩하여 특성 벡터를 생성하고, 디코더는 생성된 특성 벡터를 바탕으로 출력 시퀀스를 생성합니다. Transformer에서는 인코더와 디코더 내부에 여러 개의 층(layer)을 쌓아서 특성 벡터를 추출하고 출력을 생성합니다.

이러한 Transformer의 구조와 동작 방식은 자연어 처리 분야에서 큰 혁신을 가져왔으며, 딥러닝을 이용한 자연어 처리 분야의 연구와 산업에서 널리 활용되고 있습니다.

참고로, 최근에는 이러한 Transformer의 아키텍쳐 자체를 수정한 모델들도 발표되었었는데요. Scaling law 관점에서 Vanilla Transformer가 제일 괜찮은 성능을 보여주었다는 연구 결과도 있습니다.

![image](https://user-images.githubusercontent.com/7252598/227702765-b4d575b0-b32a-43a3-926f-ca18d7a4fe1c.png)

- Tay, Yi, et al. "Scaling Laws vs Model Architectures: How does Inductive Bias Influence Scaling?." arXiv preprint arXiv:2207.10551 (2022).
