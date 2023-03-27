# Setup
본 문서에서는 환경설정에 대한 내용을 다룰 예정입니다. 
대부분의 라이브러리는 파이썬 패키지로 구성됩니다.
개발환경은 코랩 노트북(Colab notebook), 파이썬 가상환경(virtual environment) 등 다양한 방법으로 셋팅할 수 있습니다만, 초보자의 경우 코랩 노트북 실습 환경이 접근하기 용이하다고 판단되기에 실습은 주로 코랩 노트북 기준으로 진행될 예정입니다.

코랩 노트북 환경에 아직 익숙하지 않다면, 해당 [코랩 노트북 가이드](https://colab.research.google.com/notebooks/intro.ipynb)를 참조하시기를 추천합니다.

## 구글 코랩 노트북 사용하기
코랩 환경에서는 **pip**를 통해 원하는 패키지를 설치해줍니다. pip와 같은 커맨트 명령어는 !를 앞에 붙여줌으로써 사용할 수 있습니다.

```
!pip install transformers
```

설치가 완료되면, `import transformers` 명령어로 패키지를 임포트해서 사용할 수 있습니다.


![set_env_0](https://user-images.githubusercontent.com/7252598/144546300-6a76adaa-dd2a-4b2d-826b-a5a95ed86b1a.gif)

자 이제 허깅페이스 라이브러리(transformers)를 설치완료했습니다. 하지만 자연어처리를 하기 위한 다른 여러가지 기계학습 프레임워크 (`PyTorch, TensorFlow, scikit-learn`) 설치가 남았습니다. 다음장에서는 다른 기계학습 프레임워크 및 실습환경을 위한 라이브러리 환경을 셋팅해보겠습니다.
