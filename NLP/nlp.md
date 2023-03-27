# Natural Language Processing
Transformer model에 대해서 살펴보기 전에, 자연어처리가 무엇인지에 대해서 빠르게 살펴보겠습니다.

## 자연어처리란?
자연어처리란 인간의 언어와 관련된 모든 것을 이해하는데 중점을 둔 언어학(linguistics) 및 머신러닝(machine learning) 분야입니다. 자연어처리의 목적은 단순히 단어 하나하나를 이해하는것이 아닌 단어들의 문맥을 이해하는 것을 목표로 합니다.   
자연어처리 내에는 많은 태스크들이 있지만 크게 5가지로 나누어서 생각할 수 있습니다.

- 문장 분류 문제 (Classifying whole sentences): 영화 리뷰에서 감성분석(sentiment analysis)을 한다거나, email이 스팸인지 여부를 판단(spam detection)한다거나, 문장이 문법적으로 옳은지 판단(grammar correction)한다거나, 두개의 문장이 주어졌을때 논리적으로 관계가 있는지 없는지를 판단(natural language inference)하는 문제를 다룹니다.
- 문장 내 단어 분류 문제 (Classifying each word in a sentence): 문장을 이루는 단어가 문법적으로 어떤 형태소(noun, verb, adjective)에 속하는지 분류하는 문제(Part-of-speech)나, 단어에 대해 특정 엔티티(person, location, organization)로 분류하는 개체명인식(name entity recognition)을 하는 문제를 다룹니다.
- 텍스트 컨텐츠 생성 문제 (Generating text content): 문장 내 마스킹된 단어 또는 빈칸을 채우는 문제(Masked 를 다룹니다.
- 문서에서 정답을 추출하는 문제 (Extracting an answer from a text) : 본문과 질문이 주어졌을때 질문에 해당하는 답을 본문에서 추출하는 문제(machine reading comprehension)를 다룹니다.
- 주어진 입력에 대한 새로운 문장 생성 문제 (Generating a new sentence from an input text): 글을 다른 언어로 번역(translation)하거나 요약(summarization)하는 문제를 다룹니다.

## 이러한 문제들은 왜 challenging할까?
먼저, 컴퓨터는 인간과 같은 방법으로 언어를 해석하지 않습니다. 예컨데 "나는 배고프다" 라는 문장은 인간은 매우 쉽게 이해할 수 있고, "나는 배고프다"와 "나는 슬프다" 라는 문장 또한 쉽게 구분할 수 있습니다. 하지만, 머신러닝 모델이 이러한 문제를 해결하는 것은 좀 더 어렵습니다. 언어로부터 처음부터 의미를 배워야하기 때문에 적절한 언어 처리 과정을 통해 좋은 언어표현(language representation)을 얻는 것이 필요합니다. 좋은 언어표현을 얻기 위한 연구들은 계속 진행되고 있습니다. 앞으로 본 책을 통해 어떠한 방법들이 있는지 알아보겠습니다.