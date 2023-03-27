# Transformer Family

Transformer는 기본적으로 인코더-디코더 번역 모델이지만, 인코더 혹은 디코더만 사용하거나 학습 방법을 변경해주는 방식으로 다양한 변형이 가능합니다. 아래에서는 Transformer Family의 대표적인 모델들을 간단히 설명해보겠습니다.

## Encoder Models
- 입력 문장을 인코딩하여 문맥 정보를 추출하는 구조
  
#### BERT (Bidirectional Encoder Representations from Transformers) 
구글에서 발표한 사전 학습 언어 모델
양방향 Transformer Encoder를 사용하여 문맥 정보를 양방향으로 학습
Masked Language Modeling (MLM)과 Next Sentence Prediction (NSP) 두 가지 태스크를 학습
자연어 이해와 관련된 다양한 태스크에서 좋은 성능을 보임


#### RoBERTa (A Robustly Optimized BERT Pretraining Approach)
Meta(Facebook)에서 발표한 사전 학습 언어 모델
BERT 모델을 기반으로 하지만, 하이퍼파라미터를 최적화하여 더 좋은 성능을 보임
MLM과 NSP 태스크를 학습하며, Dynamic masking을 적용하여 학습 데이터를 더욱 다양하게 만듦
다양한 자연어 처리 태스크에서 BERT보다 우수한 성능을 보임

#### ALBERT (A Lite BERT for Self-supervised Learning of Language Representations)
구글에서 발표한 사전 학습 언어 모델
BERT와 유사한 구조를 가지지만, 파라미터 수를 대폭 감소시키는 Lite 버전
다양한 학습 방법과 하이퍼파라미터 최적화를 적용하여 좋은 성능을 보이면서도 BERT보다 파라미터 수가 18배 이상 적음
대용량 모델의 한계를 극복하기 위한 방법으로 제안됨


#### ELECTRA (Efficiently Learning an Encoder that Classifies Token Replacements Accurately)
구글에서 발표한 사전 학습 언어 모델
Generator와 Discriminator라는 두 개의 네트워크를 사용하여 임의의 단어를 마스크 처리하고, Discriminator는 이를 실제 단어인지 가짜 단어인지 판별하여 학습
BERT와 같이 masking이 적용된 15%의 토큰에 대해서만 계산하는 것이 아니라 모든 토큰에 대해서 loss를 계산하므로 더 효율적인 학습이 가능함
BERT보다 파라미터 수가 적으면서도 더 높은 정확도를 보임

#### DeBERTa
TBD(작성예정)

## Decoder Models
- 인코더 모델과 비슷한 구조를 가지지만, 인코더에서 추출된 문맥 정보를 이용하여 디코딩
- 자신의 이전 출력값과 인코더에서의 출력값을 이용하여 다음 출력값 생성

#### GPT-1, GPT-2 (Generative Pre-trained Transformer) 
인코더-디코더 구조의 Transformer 모델 중 디코더 부분만 사용하여 생성 모델로 사용
주로 자연어 생성 태스크에서 사용

#### CTRL
텍스트 생성시 지정된 제어 토큰을 사용하여 조작이 가능합니다.
특정 주제, 스타일, 감정 등을 지정하여 텍스트 생성이 가능하며, 일반적인 자연어 처리 태스크에서도 뛰어난 성능을 보입니다.

#### GPT-3
175B 개의 파라미터를 가지고 있습니다.
일반적인 자연어 처리 태스크에서 뛰어난 성능을 보이며, zero-shot, one-shot, few-shot 등 다양한 학습 방식을 지원합니다.
대량의 텍스트 데이터를 사용하여 사전학습하였기 때문에, 매우 다양한 자연어 처리 태스크에서 높은 성능을 보여줍니다.

#### GPT-Neo/GPT-J-6B
EleutherAI라는 오픈소스 진영에서 만든 GPT 모델입니.

## Enocder-Decoder Models
- 인코더와 디코더를 모두 사용한 모델로써 NLU와 NLG 두가지에 모두 강점이 있는 구조

#### T5 (Text-to-Text Transfer Transformer)
Google에서 개발한 모델로, 다양한 자연어 처리 태스크에 대해 일관된 방식으로 학습 및 평가를 수행할 수 있도록 설계되었습니다. 주요 특징은 다음과 같습니다.
다양한 자연어 처리 태스크를 Text-to-Text Transfer Learning 방식으로 학습합니다. 이는 특정 태스크를 위한 별도의 모델을 학습할 필요 없이, 하나의 모델에서 다양한 태스크를 수행할 수 있게 합니다.

#### BART (Bidirectional and Auto-Regressive Transformer)
Facebook AI Research에서 개발한 모델로, 언어 생성 및 기계 번역에 사용됩니다. 주요 특징은 다음과 같습니다.
양방향 및 자동 회귀 모델: BART는 양방향 인코더와 자동 회귀 디코더를 가지고 있으며, 다양한 언어 생성 태스크에 적용할 수 있습니다.

#### BigBird
TBD(작성예정)

#### T0
TBD(작성예정)

#### UL2
TBD(작성예정)
