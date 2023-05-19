# PyTorch로 분산어플리케이션 개발하기
이번챕터에서는 LLM 학습을 위한 여러가지 개념들에 대해서 다뤄보도록 하겠습니다.

## 용어

### Node & GPU
- Node: 컴퓨터
- Rank: GPU 번호 (프로세스 번호)
- World Size: GPU 개수 (프로세스 개수)
- Global / Local: 전체시스템 범위 / 한 노드내 범위

<img width="1154" alt="image" src="https://github.com/eagle705/nlp-book/assets/7252598/82ea0e0c-b49c-48dc-ad10-b8c1ee06baf8">


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

### 분산학습에 쓰이는 통신기술
- PCI (PCIe): 한 노드 내에서의 DMA를 지원하는 버스 프로토콜
- NVLink: 한 노드 내에서 GPU간 DMA 통신 지원 (GPU간 통신 특화라 매우 빠름)
- Ethernet: 노드와 노드사이의 통신을 위한 프로토콜 (보통 RDMA안됨)
- Infiniband: 노드와 노드 사이의 RDMA를 위해 사용되는 네트워크 프로토콜 (이더넷보다 빠름)

<img width="932" alt="image" src="https://github.com/eagle705/nlp-book/assets/7252598/717f091c-fa52-4771-af5c-e363ca4c84fb">

정리하면 다음과 같고 분산환경학습에서는 NVLink(노드내에서)와 Infiniband(노드간) 두개만 알면 됩니다. 아래 그림은 비교에 대한 도표입니다.

<img width="1026" alt="image" src="https://github.com/eagle705/nlp-book/assets/7252598/fcf25fba-810f-4720-9283-5cf47820407d">



### MPI (Message Passing Interface)
메세지 패싱은 두 프로세서가 데이터를 공유하기 위한 방법중 한가지입니다. 데이터를 서로 공유하기 위해 메세지라는 객체를 만들고 객체 안에는 태그라는 데이터가 있어서 식별용도로 사용할 수 있습니다. 딥러닝에서 병렬처리에서는 보통은 메세지패싱 방법을 사용하게 됩니다.

Message Passing | Message Passing vs Shared Memory
---|---
<img width="1007" alt="image" src="https://github.com/eagle705/nlp-book/assets/7252598/6dcb6614-9ed8-4ea0-9f7b-1abe39224234"> | <img width="845" alt="image" src="https://github.com/eagle705/nlp-book/assets/7252598/517651e4-6a6b-4b97-89f4-e8dec4ecc1c8">

메세지 패싱을 사용할때 주로 쓰는 연산들이 있고 그것에 대한 인터페이스를 정의한 것이 MPI입니다. Open MPI라는 인터페이스를 구현한 오픈소스 라이브러리가 있지만 대부분의 경우 C++로 구현된 NCCL과 GLOO를 사용합니다. 이 라이브러리들은 [torch.distributed](https://pytorch.org/docs/stable/distributed.html)에 포함되어 있어서 쉽게 사용할 수 있습니다.

- NCCL: NVIDIA에서 개발한 collective communication library
  - 주로 GPU VRAM간 통신시 사용
  - Infiniband with VRAM 지원
  - Ethernet with VRAM 지원
  - CPU RAM 통신 미지원
  - GPU에서 학습할 때 주로 사용함
- GLOO:Meta에서 개발한 collective communication library
  - Infiniband에서는 IP가 사용가능해야함
  - Ethernet with RAM 지원
  - [ZeRO처럼 offloading](https://www.deepspeed.ai/tutorials/zero-offload/)해서 써야할때 유용함



#### Point-to-Point(P2P)
- 하나의 프로세스에서 다른 프로세스로 데이터를 전송하는 것
- `send` 와 `recv` 함수 또는 즉시 응답하는(immediate counter-parts) `isend` 와 `irecv` 를 사용
- `send/recv` 는 모두 블로킹 함수
  - 두 프로세스는 통신이 완료될 때까지 멈춰있음
- `isend/irecv` 는 논-블로킹 함수
  - 즉시 응답; 스크립트는 실행을 계속하고 메소드는 wait() 를 선택할 수 있는 Work 객체를 반환합니다.

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

#### Collective Communication 통신(집합통신)
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
- dist.barrier(group): group 내의 모든 프로세스가 이 함수에 진입할 때까지 group 내의 모든 프로세스를 멈춥(block)니다.
- dist.broadcast(tensor, src, group): src 의 tensor 를 모든 프로세스의 tensor 에 복사합니다. 
- dist.reduce(tensor, dst, op, group): op 를 모든 tensor 에 적용한 뒤 결과를 dst 프로세스의 tensor 에 저장합니다. 
- dist.all_reduce(tensor, op, group): 리듀스와 동일하지만, 결과가 모든 프로세스의 tensor 에 저장됩니다. 
- dist.scatter(tensor, scatter_list, src, group): i 번째 Tensor scatter_list[i] 를 i 번째 프로세스의 tensor 에 복사합니다. 
- dist.gather(tensor, gather_list, dst, group): 모든 프로세스의 tensor 를 dst 프로세스의 gather_list 에 복사합니다. dist.all_gather(tensor_list, tensor, group): 모든 프로세스의 tensor 를 모든 프로세스의 tensor_list 에 복사합니다. dist.barrier(group): group 내의 모든 프로세스가 이 함수에 진입할 때까지 group 내의 모든 프로세스를 멈춥(block)니다.


### 3D Parallelism

#### Data Parallelism




## Reference
- PyTorch 공식 튜토리얼
- https://nbviewer.org/github/tunib-ai/large-scale-lm-tutorials/blob/main/notebooks/07_tensor_parallelism.ipynb
