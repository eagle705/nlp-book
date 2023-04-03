# SLURM

Slurm은 리눅스 운영 체제에서 작업 스케줄링과 관리를 위한 오픈 소스 클러스터 관리 시스템입니다. Slurm은 Supercomputing, High Performance Computing (HPC), 및 클러스터 환경에서 많이 사용됩니다.

Slurm은 사용자가 여러 노드에서 실행되는 다수의 작업을 제출하고 추적할 수 있도록하는 작업 스케줄러입니다. 또한 노드 리소스의 사용률을 모니터링하고 시스템의 사용 가능한 자원을 최대한 활용하여 효율적으로 작업을 배치합니다.

Slurm은 사용자가 작업을 제출하고 대기열에 추가하는 데 사용되는 명령 줄 인터페이스와, 작업 및 작업 진행 상황을 모니터링하는 데 사용되는 웹 기반 사용자 인터페이스를 제공합니다. 또한 사용자가 작업을 중단하거나 삭제하거나 우선순위를 지정하는 등의 기능도 제공합니다.

Slurm은 대부분의 리눅스 배포판에서 사용할 수 있으며, 많은 HPC 클러스터 시스템에서 사용되고 있습니다. Slurm은 또한 다양한 기능과 유연성을 제공하기 때문에, 다양한 연구 분야에서 사용됩니다.

Slurm을 사용하여 간단한 예제 작업을 제출하는 방법입니다.

1. 작업 스크립트 작성   
먼저 작업 스크립트를 작성해야 합니다. 이 예제에서는 "hello_world.sh" 라는 이름의 스크립트를 사용하겠습니다. 이 스크립트는 단순히 "Hello, World!"를 출력합니다.

```bash
#!/bin/bash
echo "Hello, World!"
```

2. 작업 제출
다음으로, 작업을 제출해야 합니다. 이를 위해 다음 명령어를 사용합니다.

```bash
sbatch hello_world.sh
```

이 명령은 "hello_world.sh" 파일을 제출하고 작업 ID를 반환합니다.

3. 작업 상태 확인
작업이 제출되면 현재 작업 상태를 확인할 수 있습니다. 이를 위해 다음 명령어를 사용합니다.

```bash
squeue -u [사용자명]
```

위 명령어는 현재 대기열에서 실행 중인 모든 작업을 보여줍니다. 작업이 실행되면 작업 ID와 함께 상태 정보가 표시됩니다.

4. 작업 결과 확인
작업이 완료되면 결과를 확인할 수 있습니다. 이를 위해 작업이 실행되는 노드의 로그 파일을 확인합니다.

```bash
less [작업ID].out
```

위 명령어는 작업이 출력한 결과를 볼 수 있습니다.

이렇게 Slurm을 사용하여 간단한 작업을 제출하고 결과를 확인할 수 있습니다. Slurm은 이러한 기능 외에도 다양한 옵션을 제공하므로 더 복잡한 작업도 처리할 수 있습니다.

대형언어모델은 다음과 같은 예시 스크립트로 실행할 수 있습니다.
- 스크립트(job.slurm)
```bash
#!/bin/bash
#SBATCH --job-name=megatronlm    # 작업 이름
#SBATCH --nodes=4                # 사용할 노드 수
#SBATCH --ntasks-per-node=8      # 각 노드에서 실행할 태스크 수
#SBATCH --cpus-per-task=10       # 각 태스크에서 사용할 CPU 수
#SBATCH --time=24:00:00          # 예상 실행 시간
#SBATCH --mem=0                  # 사용할 메모리 양 (0은 제한 없음)
#SBATCH --partition=gpu          # 사용할 파티션 이름
#SBATCH --gres=gpu:8             # 사용할 GPU 수

module load cuda/11.4
module load nccl/2.9

mpirun -n $SLURM_NTASKS --npernode $SLURM_NTASKS_PER_NODE \
python train.py \
--model-parallel-size 2 \
--num-layers 12 \
--hidden-size 3072 \
--batch-size 8 \
--seq-length 1024 \
--lr 0.00015 \
--lr-decay-iters 320000 \
--lr-decay-style cosine \
--min-lr 0.0 \
--warmup .01 \
--optimizer adamw \
--adam-eps 1e-06 \
--weight-decay 0.1 \
--clip-grad 1.0 \
--skip-invalid-size-inputs-valid-test \
--log-interval 10 \
--save-interval 1000 \
--eval-interval 1000 \
--tensor-model-parallel-size 1 \
--pipeline-model-parallel-size 1 \
--distributed-backend nccl \
--deepspeed
```

- 작업 제출
`sbatch job.slurm`
