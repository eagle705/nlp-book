# Contextual word representation

Contextual word representation(문맥적 단어 표현)은 자연어 처리 분야에서 최근에 주목받는 연구 주제 중 하나입니다. 이는 단어를 벡터로 표현하는 Word2Vec, GloVe와 같은 이전의 모델들과 달리, 단어의 의미를 문맥과 함께 고려하여 표현합니다.

ELMo(Embeddings from Language Models)는 Contextual word representation을 제공하는 대표적인 모델 중 하나입니다. ELMo는 양방향 LSTM(BiLSTM) 모델을 기반으로 하며, 각 단어의 표현을 생성하기 위해 해당 단어의 주변 단어와 문장 전체의 정보를 모두 고려합니다. 즉, ELMo 모델은 단어의 의미를 문맥과 함께 파악할 수 있도록 합니다.

BERT(Bidirectional Encoder Representations from Transformers)는 ELMo와 유사하게 양방향 Transformer 모델을 사용하여 문맥적 단어 표현을 생성하는 모델입니다. BERT 모델은 사전 학습된 언어 모델을 활용하여 다양한 자연어 처리 태스크에 대해 높은 성능을 보여주고 있습니다.

GPT(Generative Pre-trained Transformer)는 BERT와 유사한 Transformer 모델을 사용하지만, 단어를 생성하는 언어 모델을 학습합니다. GPT는 말뭉치에서 학습된 언어 모델을 이용하여 문장 생성, 기계 번역 등 다양한 태스크에 활용됩니다.

최근에는 이러한 Contextual word representation을 사용하여 다양한 자연어 처리 태스크에서 높은 성능을 보이는 모델들이 연구되고 있습니다. 예를 들어, 자연어 이해, 질문 응답, 감성 분석, 기계 번역 등의 태스크에서 Contextual word representation을 활용한 모델들이 좋은 성능을 보여주고 있습니다.
