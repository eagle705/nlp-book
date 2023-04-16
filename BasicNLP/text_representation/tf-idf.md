# TF-IDF

TF-IDF는 자연어 처리에서 널리 사용되는 단어의 가중치를 나타내는 방법 중 하나입니다. 이 방법은 특정 문서에서 어떤 단어가 중요한지를 계산하는 데 사용됩니다.

TF-IDF는 두 가지의 개념인 TF(Term Frequency)와 IDF(Inverse Document Frequency)를 합쳐서 계산됩니다.

Term Frequency (TF)
TF란 문서에서 단어가 나타난 횟수를 나타내는 값입니다. 예를 들어, 문서에서 "cat"이라는 단어가 5번 나타난다면 "cat"의 TF는 5가 됩니다.

Inverse Document Frequency (IDF)
IDF란 단어의 중요도를 나타내는 값입니다. 단어가 특정 문서에서 자주 나타난다면 해당 단어는 중요하지 않을 가능성이 높습니다. 그러나 여러 문서에서 자주 나타난다면 해당 단어는 전체적으로 중요한 단어일 가능성이 높습니다. IDF는 이러한 개념을 나타내는데 사용됩니다. IDF 값은 전체 문서 수를 해당 단어가 포함된 문서 수로 나눈 다음 로그를 취한 값입니다.

TF-IDF
TF-IDF는 각 단어의 TF와 IDF를 곱해서 계산됩니다. 즉, TF가 높을수록 해당 단어가 문서에서 중요하다는 것을 나타내고, IDF가 높을수록 해당 단어가 전체 문서에서 중요하다는 것을 나타냅니다.

예를 들어, "cat"이라는 단어가 문서에서 5번 나타난 경우, "cat"의 TF는 5입니다. 이 단어가 10,000개의 문서 중 1,000개의 문서에 나타난 경우, "cat"의 IDF는 log(10,000/1,000) = 1 입니다. 따라서 "cat"의 TF-IDF 값은 5 * 1 = 5가 됩니다.

TF-IDF는 문서의 유사도 측정, 검색 엔진에서의 검색 결과 랭킹 등에서 활용됩니다.

$$\mathrm{tf}(t,d) = \frac{\text{단어 t가 문서 d에서 나타난 빈도}}{\text{문서 d의 총 단어 수}}$$

$$\mathrm{idf}(t, D) = \log{\frac{\text{총 문서 수}}{\text{단어 t가 나타난 문서 수} + 1}}$$

$$\mathrm{tf\text{-}idf}(t, d, D) = \mathrm{tf}(t,d) \cdot \mathrm{idf}(t, D)$$

여기서 $t$는 단어, $d$는 문서, $D$는 전체 문서 집합을 나타냅니다.