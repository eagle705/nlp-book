# PyTorch로 분산어플리케이션 개발하기
이번 챕터에서는 LLM 학습을 위한 여러가지 개념들에 대해서 다뤄보도록 하겠습니다.   
딥러닝 모델이 커지고 학습해야될 데이터양이 많아지다보면 데이터와 모델을 여러 GPU에 그리고 여러 노드에 나눠서 학습해야할 일이 생기게 됩니다. 먼저 용어부터 살펴보도록 하겠습니다.

## 용어

### Node & GPU
- Global / Local: 전체시스템 범위 / 한 노드내 범위
- Node: 컴퓨터
  - node_size: 독립된 머신의 수 (컴퓨터의 수)
- Rank: GPU 번호 (글로벌 프로세스 번호)
  - local_rank: 해당 node에서의 프로세스 번호
- num_gpu: 각 노드당 사용하는 GPU 개수
- World Size: GPU 개수 (총 글로벌 프로세스 개수이며 node_size * num_gpu)


<img width="1154" alt="image" src="https://github.com/eagle705/nlp-book/assets/7252598/82ea0e0c-b49c-48dc-ad10-b8c1ee06baf8"> 

Global Rank | Local Rank
---|---
 ![image](https://github.com/eagle705/nlp-book/assets/7252598/34d654d4-1cc8-44a2-b7eb-322fe55e3899) | ![image](https://github.com/eagle705/nlp-book/assets/7252598/4d8bad15-d294-48fb-87d3-9d8c721c88f4)


### HW
- PIO (programmed IO)
  - 보통 사용자가 파일을 읽고 쓰는 경우
- DMA (Direct Memory Access)
  - 바로 디스크에서 읽어오는 경우
- RDMA (Remote Direct Memory Access)
  - 네트워크 통신을 통해 다른 노드에 있는 디스크에서 DMA로 읽어오는 경우

PIO, DMA | RDMA
---|---
<img width="572" alt="image" src="https://github.com/eagle705/nlp-book/assets/7252598/bad85369-cdc2-4025-93f0-62b30d713189"> | <img width="807" alt="image" src="https://github.com/eagle705/nlp-book/assets/7252598/e34aa3cc-a98c-45a8-a568-db481cc589ae">

## 분산학습에 쓰이는 통신기술
- PCI (PCIe): 한 노드 내에서의 DMA를 지원하는 버스 프로토콜
- NVLink: 한 노드 내에서 GPU간 DMA 통신 지원 (GPU간 통신 특화라 매우 빠름)
- Ethernet: 노드와 노드사이의 통신을 위한 프로토콜 (보통 RDMA안됨)
- Infiniband: 노드와 노드 사이의 RDMA를 위해 사용되는 네트워크 프로토콜 (이더넷보다 빠름)

<img width="932" alt="image" src="https://github.com/eagle705/nlp-book/assets/7252598/717f091c-fa52-4771-af5c-e363ca4c84fb">

정리하면 다음과 같고 분산환경학습에서는 NVLink(노드내에서)와 Infiniband(노드간) 두개만 알면 됩니다. 아래 그림은 비교에 대한 도표입니다.

<img width="1026" alt="image" src="https://github.com/eagle705/nlp-book/assets/7252598/fcf25fba-810f-4720-9283-5cf47820407d">


## MPI (Message Passing Interface)
MPI는 HPC에서 분산 시스템 구현을 위해 1990년대 초에 만들어진 표준화된 데이터 통신 라이브러리입니다.
여기서 나오는 메세지 패싱은 두 프로세서가 데이터를 공유하기 위한 방법중 한가지입니다. 데이터를 서로 공유하기 위해 메세지라는 객체를 만들고 객체 안에는 태그라는 데이터가 있어서 식별용도로 사용할 수 있습니다. 딥러닝에서 병렬처리에서는 보통은 메세지패싱 방법을 사용하게 됩니다.

Message Passing | Message Passing vs Shared Memory
---|---
<img width="1007" alt="image" src="https://github.com/eagle705/nlp-book/assets/7252598/6dcb6614-9ed8-4ea0-9f7b-1abe39224234"> | <img width="845" alt="image" src="https://github.com/eagle705/nlp-book/assets/7252598/517651e4-6a6b-4b97-89f4-e8dec4ecc1c8">

메세지 패싱을 사용할때 주로 쓰는 연산들이 있고 그것에 대한 인터페이스를 정의한 것이 MPI입니다. Open MPI라는 인터페이스를 구현한 오픈소스 라이브러리가 있지만 대부분의 경우 C++로 구현된 NCCL과 GLOO를 사용합니다. 이 라이브러리들은 [torch.distributed](https://pytorch.org/docs/stable/distributed.html)에 포함되어 있어서 쉽게 사용할 수 있습니다.

![image](https://github.com/eagle705/nlp-book/assets/7252598/9ce68cf4-4058-41e5-a8e8-ce8d6368127b)
- NCCL(NVIDIA Collective Communications Library; Nickel로 발음): NVIDIA에서 개발한 collective communication library입니다. GPU 간 통신은 NVLink, PCIe, GPU Direct P2P를 통해 통신하며 노드 간에는 Socket, Infiniband를 통해 통신합니다. 
  - 주로 GPU VRAM간 통신시 사용
  - Infiniband with VRAM 지원
  - Ethernet with VRAM 지원
  - CPU RAM 통신 미지원
  - GPU에서 학습할 때 주로 사용함
- GLOO:Facebook(Meta)에서 개발한 collective communication library 대부분의 Linux에서 사용 가능하며 CPU 병렬화시에 권장하고 있습니다.
  - Infiniband에서는 IP가 사용가능해야함
  - Ethernet with RAM 지원
  - [ZeRO처럼 offloading](https://www.deepspeed.ai/tutorials/zero-offload/)해서 써야할때 유용함



### Point-to-Point(P2P)
- 하나의 프로세스에서 다른 프로세스로 데이터를 전송하는 것
- `send` 와 `recv` 함수 또는 즉시 응답하는(immediate counter-parts) `isend` 와 `irecv` 를 사용
- `send/recv` 는 모두 블로킹 함수
  - 두 프로세스는 통신이 완료될 때까지 멈춰있음
- `isend/irecv` 는 논-블로킹 함수
  - 즉시 응답함. 스크립트는 실행을 계속하고 메소드는 wait() 를 선택할 수 있는 Work 객체를 반환함

![image](https://github.com/eagle705/nlp-book/assets/7252598/640e12dd-0368-46ba-9243-82e5df023849)

```python
"""블로킹(blocking) 점-대-점 간 통신"""

def run(rank, size):
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0])
```

```python
"""논-블로킹(non-blocking) 점-대-점 간 통신"""

def run(rank, size):
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        req = dist.isend(tensor=tensor, dst=1)
        print('Rank 0 started sending')
    else:
        # Receive tensor from process 0
        req = dist.irecv(tensor=tensor, src=0)
        print('Rank 1 started receiving')
    req.wait()
    print('Rank ', rank, ' has data ', tensor[0])
```

### Collective Communication 통신(집합통신)
점-대-점 간 통신과 달리 집합 통신은 그룹 의 모든 프로세스에 걸친 통신 패턴을 허용합니다. 그룹은 모든 프로세스의 부분 집합입니다. 그룹을 생성하기 위해서는 dist.new_group(group) 에 순서(rank) 목록을 전달합니다. 기본적으로, 집합 통신은 월드(world) 라고 부르는 전체 프로세스에서 실행됩니다. 예를 들어, 모든 프로세스에 존재하는 모든 Tensor들의 합을 얻기 위해서는 dist.all_reduce(tensor, op, group) 을 사용하면 됩니다.

```python
""" All-Reduce 예제 """
def run(rank, size):
    """ 간단한 집합 통신 """
    group = dist.new_group([0, 1])
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print('Rank ', rank, ' has data ', tensor[0])
```


<img width="952" alt="image" src="https://github.com/eagle705/nlp-book/assets/7252598/630e6f86-211b-48ad-b4e6-f02008fe5376">


기본 연산은 다음과 같이 정리할 수 있습니다.
- broadcast: 복사하기
  - 특정 rank에 있는 데이터를 다른 통신에 참여하는 모든 rank에 복사
- scatter: 쪼개기
  - 특정 rank에서 통신에 참여하는 프로세스들에게 데이터를 쪼개서 나눠줌
- gather: 모으기
  - 여러개의 프로세스에서 갖고 있는 데이터를 모아서 하나의 rank에서 리스트형식으로 조합
- reduction: 연산하기
  - 똑같이 모으지만 리스트형식으로 쌓는게 아니라 sum, product, max, min등 연산을 통해서 값을 모으는 방법
- All-XXX: 통신 결과를 모든 모든 rank가 갖게 만드는 것
- Reduce-Scatter: 연산한 뒤 결과를 쪼개서 전달

<img width="1430" alt="image" src="https://github.com/eagle705/nlp-book/assets/7252598/bab97630-00d5-433a-8fc5-0f08bcc87884">

함수형태로 정리하면 다음과 같습니다.

- dist.broadcast(tensor, src, group): src 의 tensor 를 모든 프로세스의 tensor 에 복사합니다. 
- dist.reduce(tensor, dst, op, group): op 를 모든 tensor 에 적용한 뒤 결과를 dst 프로세스의 tensor 에 저장합니다. 
- dist.all_reduce(tensor, op, group): 리듀스와 동일하지만, 결과가 모든 프로세스의 tensor 에 저장됩니다. 
- dist.scatter(tensor, scatter_list, src, group): i 번째 Tensor scatter_list[i] 를 i 번째 프로세스의 tensor 에 복사합니다. 
- dist.gather(tensor, gather_list, dst, group): 모든 프로세스의 tensor 를 dst 프로세스의 gather_list 에 복사합니다. 
- dist.all_gather(tensor_list, tensor, group): 모든 프로세스의 tensor 를 모든 프로세스의 tensor_list 에 복사합니다. 
- dist.barrier(group): group 내의 모든 프로세스가 이 함수에 진입할 때까지 group 내의 모든 프로세스를 멈춥(block)니다.


## 3D Parallelism

지금부터는 위에서 이야기했던 Collective Communication을 활용한 다양한 병렬화 기법에 대해서 다뤄보도록 하겠습니다. 병렬화 기법은 크게 3가지가 대표적으로 Data Parallelism(DP), Model Parallelism(MP)(혹은 Tensor Parallelism(TP)), Pipeline Parallelism(PP)이 있습니다. 아래 예시는 32장의 GPU로 DP=2, PP=4, TP=4로 분할한 경우입니다.

<img width="944" alt="image" src="https://github.com/eagle705/nlp-book/assets/7252598/bc401b1f-41d9-4cd8-b533-39e5c255ed94">

### Data Parallelism (DP)

##### DP 구현
초기 DP의 경우 `torch.nn.DataParallel`에 구현되어 있습니다. 
- 싱글 프로세스 + 멀티쓰레드로 구현됨
- 싱글 프로세스이므로 멀티노드에서 사용할 수 없고 한대의 컴퓨터에서만 사용할 수 있었음


<img width="1192" alt="image" src="https://github.com/eagle705/nlp-book/assets/7252598/420b90ff-04f5-4e7c-abc3-3ecac4ddb5a0">

DP에서 Forward pass는 다음과 같이 일어납니다. GPU가 0~3까지 있다고 할때 데이터를 scatter 연산을 통해 분배해줍니다. 그 후 동일한 모델을 GPU-0에 올려놓고 GPU 1,2,3에 broadcast 연산을 통해 동일하게 복사를 해줍니다. 그 후 scatter로 분배된 각각의 데이터에 대해서 Forward를 해주고 logits을 계산하게 됩니다. 각각의 GPU에서 구한 logits을 다시 GPU-0에 gather 연산을 통해 모아주면서 모아진 값을 기반으로 기존에 갖고 있던 label과 비교해서 loss를 구하게 됩니다.
<img width="1603" alt="image" src="https://github.com/eagle705/nlp-book/assets/7252598/c9ec9445-b9bc-420a-ab87-a7925dddd7eb">

Backward pass에서는 전체 데이터에 대한 loss를 먼저 다시 scatter 연산을 통해 각각의 GPU에 분배를 해주고, 분배된 loss를 기반으로 Backward 연산을 통해 Gradient 값들을 계산해줍니다. 각각의 GPU에서 계산된 Gradient를 다시 GPU-0으로 모아서 평균을 계산하고 GPU-0에 있는 모델의 파라미터를 업데이트하게 됩니다. 업데이트 후에 다시 다음 batch에 대해서 Forward를 수행할때 broadcast 연산으로 모델을 각각의 GPU에 복사해주면서 Data Parallelism이 업데이트된 모델에 대해서 수행되게 됩니다.

<img width="1195" alt="image" src="https://github.com/eagle705/nlp-book/assets/7252598/485864b5-e5dd-42d1-b63d-e4975c142d46">

실제로 Forward와 Backward를 비교해보면 Backward의 계산량이 더 큰 편입니다.

<img width="1125" alt="image" src="https://github.com/eagle705/nlp-book/assets/7252598/62f1dde8-8cff-4354-bc21-8c7baba56f90">

이러한 DP의 문제점은 파이썬에서는 GIL(Global Interpreter Lock) 때문에 멀티쓰레드가 효율적이지 않다는 점과 노드간 통신이 불가능하다는 점입니다.  (GIL은 CPU 동작에서 적용되므로 스레드가 I/O 작업을 실행하는 동안에는 다른 스레드가 CPU 동작을 동시에 실행할 수 있기 떄문에 CPU 동작이 비교적 많지 않고 I/O 동작이 더 많은 프로그램에서는 파이썬에서도 멀티스레드가 효과가 있습니다)
![image](https://github.com/eagle705/nlp-book/assets/7252598/c2e0ca56-4b44-451d-906c-5ac214c98313)

뿐만아니라 한 디바이스(GPU-0)에서만 모델 업데이트가 되는 구조이므로 모델을 매번 복사해줘야되는 추가 비용이 있습니다. 모델이 크면 클 수록 더 큰 통신 비용이 발생하게 되는 구조입니다.

<img width="461" alt="image" src="https://github.com/eagle705/nlp-book/assets/7252598/2cc2ef6d-dc45-4f62-8973-04d8bef55cc3">

이러한 DP 구조를 개선하기 위해서는 각각의 문제를 아래와 같이 개선할 수 있습니다.
- 멀티쓰레드 기반의 구조를 멀티프로세스로 변경해서 개발하는 방법
- 한 디바이스의 모델만 업데이트 후 복사하는 방법이 아닌 모든 디바이스에서 계산된 Gradient들의 평균을 계산해서 모든 디바이스의 모델을 업데이트하는 방법 (All-reduce?)

멀티프로세스 | All-Reduce로 Gradient 평균계산
---|---
<img width="1033" alt="image" src="https://github.com/eagle705/nlp-book/assets/7252598/044eb4d3-bffb-44a3-abd8-330a79507948"> | <img width="839" alt="image" src="https://github.com/eagle705/nlp-book/assets/7252598/d7da4cfa-540b-4971-adf2-dd156616274a">

All-reduce를 통해 업데이트하는 경우 속도가 느린편이기 때문에 2017년 Baidu에서 개발한 Ring All-Reduce를 사용할 수 있습니다. (Uber의 Horovod, NCCL에서 활용되고 TensorFlow, PyTorch(DDP) 등에도 구현)

![ring_allreduce](https://github.com/eagle705/nlp-book/assets/7252598/38ad9c47-4afc-45d4-93a3-0e75ee131997)

- Ring All-Reduce 추가 설명 (TBD)
  - NCCL 실제 구현에서는 Reduce-Scatter + All-Gather와 동일
  - All-Reduce를 통해서 보내게되면 각각의 Gradient를 모두 보내서 리스트안에 담아서 더하는 구조기때문에 통신량이 Gradient의 개수가 GPU 개수만큼으로 많지만 Ring All-Reduce에서는 보내기전에 이미 더하고 그 더한값을 보내는 방식이라 보내는 데이터의 개수자체가 좀 더 적은 방식
  - [참고 예정](https://housekdk.gitbook.io/ml/ml/distributed-training/data-parallelism-overview#ring-all-reduce)

이러한 방법들을 채택해서 개발된 것이 DDP (`torch.nn.parallel.DistributedDataParallel`)입니다.
정리한 그림은 아래와 같습니다. 하지만 실제로는 뒤쪽 레이어의 Gradient부터 차례대로 All-Reduce를 진행하며 통신과 Backward를 동시에 수행해서 더 빠르게 계산하는 구조로 진행됩니다.
<img width="1299" alt="image" src="https://github.com/eagle705/nlp-book/assets/7252598/cfc96285-d497-4377-bb45-417776ff5cc1">

속도를 비교하면 다음과 같이 더 빨라지는걸 볼 수 있습니다.
![image](https://github.com/eagle705/nlp-book/assets/7252598/a8db7b97-7ec6-43bc-8488-128989858c0c)



### Model Parallelism 
Model Parallelism은 모델을 여러 GPU로 분할하는 기법을 의미합니다. 하나의 GPU에 올리기에는 큰 모델들에 대해서 MP를 적용하게 됩니다. 이때 학습하는 데이터는 모든 GPU에서 동일합니다.

<img width="1581" alt="image" src="https://github.com/eagle705/nlp-book/assets/7252598/0b4a1257-aa1b-4b9e-ad24-d2e70a53a061">

#### 2가지 MP 기법
- Inter-layer 모델 병렬화
  - 레이어 단위로 모델을 병렬화 하는 기법
- Intra-layer 모델 병렬화
  - 행렬 단위로 나눠서 병렬화하는 기법 (ex) 512 dim -> 256, 256 dim)
  - [NVIDIA의 Megatron-LM 논문](https://arxiv.org/abs/1909.08053)에 자세히 기록되어있음

<img width="1550" alt="image" src="https://github.com/eagle705/nlp-book/assets/7252598/85b0404c-38e1-439e-aebd-29671d07031c">

#### Inter-layer 기법의 단점
<img width="999" alt="image" src="https://github.com/eagle705/nlp-book/assets/7252598/7ab47436-b191-4046-899b-e0df9bc870bb">
Inter-Layer 모델 병렬화의 경우에는 배치 입력값이 인공신경망을 레이어별로 통과하는동안 이미 통과한 레이어에서는 더 이상 GPU 연산이 일어나지 않게 되는 idle 상태가 발생한다는 비효율이 존재하는데요. 이러한 비효율을 극복하기 위한 방법으로 Pipeline Parallelism이 등장하게 되었습니다.

### Pipeline Parallelism (PP)
위에서 언급한 단점을 극복하기 위해 GPU가 처리하는 배치(ex. 64 bsz)를 마이크로 배치(ex. 8 bsz)라는 더 작은 단위로 쪼개서 입력으로 넣게됩니다. 마이크로 배치의 크기를 적당한 숫자로 셋팅해주게 되면 더 빠르고 효율적인 GPU 자원활용이 가능합니다.

<img width="1265" alt="image" src="https://github.com/eagle705/nlp-book/assets/7252598/07a4a4cc-d1ba-4d60-92ef-4ef59c0ebae8">

<img width="1248" alt="image" src="https://github.com/eagle705/nlp-book/assets/7252598/56de34ff-5434-486c-9a93-4d7fe826e2bc">

하지만 위와 같은 형태의 PP 또한 단점이 있는데요. 마이크로배치로 쪼개서 효율을 추구했지만, 그 다음 배치가 입력으로 들어올때 GPU가 쉬는 bubble이 존재한다는 것 그리고 모든 마이크로배치에 대해 forward 후 backward 할때까지 Activation memory를 유지해야한다는 단점이 있습니다.

여기서 잠깐! Activation Memory가 뭘까요?
실제 backward를 하기위해서는 forward 하는 시점에 값들을 저장해놔야합니다. 실제 파이토치의 autograd.Function의 구현을 보면 아래와 같이 값을 저장하고 있습니다.
<img width="1417" alt="image" src="https://github.com/eagle705/nlp-book/assets/7252598/08c35441-a6a6-41b0-821e-44a0a5dff920">

Activation memory를 줄이는 방법은 1F1B(1 Forward, 1 Backward)라는 방식을 사용하는 것입니다. 모든 마이크로배치가 끝난 후 Backward를 하는게 아니라 마이크로배치단위로 backward를 바로 실행해주면 Activation memory를 줄일 수 있습니다. 이런 방법을 사용하면 대략 2배정도까지 메모리를 아낄 수 있습니다. PipeDream-Flush가 대표적이며 DeepSpeed, Megatron-LM등 유명한 프레임워크에서도 사용하고 있습니다. 

<img width="638" alt="image" src="https://github.com/eagle705/nlp-book/assets/7252598/e264fdb9-59cb-433a-a8e2-1cdb5fbe58c3">


#### Intra-layer 기법 알아보기
Intra layer 병렬화는 Tensor Model Parallelism (TP)이라고도 부릅니다. 행렬곱 연산의 성질을 이용하면 병렬화 전후의 연산결과를 동일하게 만들 수 있습니다.

#### Column Parallelism
입력 데이터 X는 GPU마다 복사해주고, 입력에 곱해지는 모델 파라미터 행렬 W를 수직(column)으로 쪼갠 뒤 곱해준후 다시 concat해줌으로써 쪼개기 전과 동일한 결과 값을 얻을 수 있습니다.

<img width="1305" alt="image" src="https://github.com/eagle705/nlp-book/assets/7252598/49e9086b-cb2f-4904-9ecb-6765c0bbc3b4">


#### Row Parallelism
입력 데이터는 수직(column)으로 쪼개고 모델 파라미터 행렬 W는 수평(row)로 쪼갠 뒤 더해주는 방식으로 진행하면 쪼개기 전과 동일한 결과 값을 얻을 수 있습니다.
<img width="1315" alt="image" src="https://github.com/eagle705/nlp-book/assets/7252598/a6c7f75f-339d-4d54-87b2-99fa00d92ad0">


#### MLP 연산
MLP의 경우엔 중간에 Activation이 있기 때문에 Addition이 들어가지 않는 Column Parallelism을 먼저 사용하게 되고 그 결과를 바로 활용할 수 있는 Row Parallelism을 이어 붙임으로써 빠르게 구현할 수 있습니다
Column -> Row 순으로 연결하면 빠르게 구현이 가능함 (All-gather와 Scatter 연산 생략 가능)

<img width="1555" alt="image" src="https://github.com/eagle705/nlp-book/assets/7252598/a2cfbff9-d8fb-439f-a882-553171d00fe8">

![image](https://github.com/eagle705/nlp-book/assets/7252598/919a011f-d320-4818-bcf8-9bb1a1d168ec)


#### Attention Layer 연산
Attention의 경우에도 QKV는 Column으로 쪼개고 dropout쪽은 Row로 쪼개서 진행할 수 있습니다.

#### Embedding Layer 연산
임베딩의 경우 vocab size를 기준으로 병렬화를 진행합니다. 입력데이터가 들어왔을때 각 GPU에서 처리할 수 있는 vocab에 대해서만 처리하고 나머지는 MASK 처리를 하게 됩니다. 병렬처리 되어 있는 Embedding Layer에서 각각의 vocab index에 대한 Embedding을 꺼내온 후 All-reduce (sum)을 하게 되면 원래 의도했던 vocab index와 대응되는 전체 Embedding matrix를 얻을 수 있게 됩니다.
<img width="1364" alt="image" src="https://github.com/eagle705/nlp-book/assets/7252598/392fce65-f539-4e25-be10-3b373182117f">



## Gradient Accumulation
위와 같은 다양한 Parallism 기법을 적용해도 GPU 메모리가 부족해서 batch size를 키우기 어려운 경우가 생깁니다. 즉 우리가 원하는 만큼의 데이터를 한번에 학습하기 어려운 경우가 생깁니다. 이런 경우에는 매 스텝마다 모델의 파라미터를 업데이트하지않고 parameter.grad 텐서에서 여러 backward 연산의 gradient들을 모았다가 일정 스텝이 지나면 파라미터를 업데이트하는 방법이 있습니다. 이를 통해 큰 batch size로 모델의 파라미터를 업데이트하는 효과를 낼 수 있으며 이러한 기법을 Gradient Accumulation이라 부릅니다.

![image](https://github.com/eagle705/nlp-book/assets/7252598/da4ac170-c3f3-49e5-8905-9b06bf8409db)

이때 loss도 합산되므로 `accumulation_steps`으로 나누어 주어야합니다.

```python
model.zero_grad()                                   # Reset gradients tensors
for i, (inputs, labels) in enumerate(training_set):
    predictions = model(inputs)                     # Forward pass
    loss = loss_function(predictions, labels)       # Compute loss function
    loss = loss / accumulation_steps                # Normalize our loss (if averaged)
    loss.backward()                                 # Backward pass
    if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
        optimizer.step()                            # Now we can do an optimizer step
        model.zero_grad()                           # Reset gradients tensors
        if (i+1) % evaluation_steps == 0:           # Evaluate the model when we...
            evaluate_model()                        # ...have no gradients accumulated
```

## Gradient Checkpointing
메모리가 부족한 경우, 속도는 조금 느려지더라도 메모리 사용량을 더 확보하기 위해 적용하는 기법중 하나이며 모든 노드에 gradient를 저장하지 않고 일부 노드에만 gardient를 저장함하는 기법이다.
- Ref: https://only-wanna.tistory.com/entry/Gradient-checkpointing%EC%9D%B4%EB%9E%80

## NVIDIA Apex
- TBD


## Reference
- PyTorch 공식 튜토리얼
- https://nbviewer.org/github/tunib-ai/large-scale-lm-tutorials/blob/main/notebooks/07_tensor_parallelism.ipynb
- https://velog.io/@miracle-21/PYTHONGIL-Multi-Thread
- https://housekdk.gitbook.io/ml/ml/distributed-training/data-parallelism-overview
- https://algopoolja.tistory.com/95
- https://only-wanna.tistory.com/entry/Gradient-checkpointing%EC%9D%B4%EB%9E%80
