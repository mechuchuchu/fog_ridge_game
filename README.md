# 안개 능선

[룰북 v0.5](fog-ridge-rulebook.md)와 [RL 설계 v0.2](fog-ridge-rl-design.md)를 구현하는 Python 연구 프로젝트다. 현재는 **솔로 PPO와 교신을 포함한 비동기 다인 MAPPO**를 구현했다. 이탈·아카이브와 GPU 환경 벡터화는 후속 단계다.

```bash
uv sync
uv run pytest -q
uv run fog-ridge --episodes 30 --dimension 16 --duration 60 --output results/solo-smoke
```

`results/solo-smoke/summary.json`에 설정, 시드별 초기·최종 gap, 최종 log10 gap의 사분위수와 행동 횟수가 저장된다. `events.jsonl`에는 행동 완료 또는 시간 종료 시점의 소요 시간, 완료 여부, 보상, 진짜 점수가 저장된다. 이 파일은 평가용 특권 로그이며 actor 입력이 아니다. 기록장 관리 행동은 시간이 0이며 현재 이벤트 로그에는 포함하지 않는다.

구현한 내용:

- 100개의 이동된 2차 맵, 로그 균등 대각 곡률, 폐형식 평균 고도와 gap.
- 연속시간 측정(`B*m`), 점수계(`3m`), 걸음(`m/10`), 대기.
- 현재 맵을 포함한 중복 없는 B장 측정. 현재 맵은 완료된 걸음에서만 다시 뽑는다. 0거리 걸음도 맵을 다시 뽑으며, 같은 맵이 다시 뽑힐 수도 있다.
- 최대 걸음 길이 clipping, 5초 주기 대시보드, 제한 시간에 끝나지 않은 행동 폐기.
- 유한 기록장, 삭제와 방향 기록 요약, 가득 찼을 때 새 기록을 대기시킨 후 강제 슬롯 관리.
- 좌표축을 노출하지 않는 노름·코사인 관측과 기록장 방향·위치 가중합 행동 디코더.
- 잠재함수 `-log10(gap + 1e-12)` 차이 보상, 실제 결정 간격 `info['dt']`.
- 관측만 사용해 행동을 고르는 감쇠 SGD, 주기적인 점수계 사용, 시드별 평가 CLI.

Python에서 직접 실행할 수도 있다.

```python
from fog_ridge import Config, SoloEnv
from fog_ridge.baseline import DecayingSGD, rollout

env = SoloEnv(Config(dimension=16, duration=60), seed=42)
result = rollout(env, DecayingSGD(learning_rate=0.08))
print(result)
```

`step()`은 `(observation, reward, done, info)`를 반환한다. 불법 행동은 상태를 변경하기 전에 `ValueError`로 거절한다. `info`, `env.landscape`, `env.position`, `env.records`는 시뮬레이터·평가용 인터페이스다. 정책에는 `observe()`의 결과와 `compose_walk()` 디코더만 전달한다. PPO 네트워크는 아래 학습 명령으로 사용할 수 있다. Gymnasium 연동은 아직 없다.

동시각 처리는 **행동 완료 → 대시보드 → 종료** 순서다. 정확히 T에 끝나는 행동은 완료되며, T를 넘기는 행동은 폐기된다. 측정 중에 지난 대시보드 갱신은 직전 측정값을 사용한다. 현재 솔로 구현은 결정을 건너뛰며 대시보드 사건을 처리하므로 별도의 heap 이벤트 큐가 필요하지 않다.

기록장 요약은 방향이 있는 두 기록에만 허용한다. 위치와 방향은 평균, 시각은 최신값이고 고도·점수·B는 합치지 않는다. 점수계 기록은 삭제할 수 있다. 끝나는 순간 가득 찬 기록장은 최종 점수에 영향을 주지 않으므로 추가 관리 결정을 요구하지 않는다.

맵 잡음의 `spread`는 차원이 바뀌어도 전형적인 중심 이동 거리가 비슷하도록 좌표별 표준편차를 `spread / sqrt(D)`로 사용한다. 시작 위치는 실제 100맵 평균 중심에서 `start_radius`만큼 떨어져 있고, 기본 `S_max`는 그 거리의 1/3이다. 세계 생성 난수와 행동에 따른 샘플링 난수를 분리했다.

현재 베이스라인은 학습이나 튜닝을 수행하지 않는다. B는 고정이고 학습률은 `lr / (1 + decay * steps)`다. 최고 점수계 기록을 가능한 한 보존하지만 막판 복귀는 아직 구현하지 않았다. 관측의 점수계·대시보드 값도 현재 원시값이며, 학습 단계에서 상대값 정규화를 추가해야 한다.

다음 구현 단계는 교신 정책 튜닝과 커리큘럼 평가, 이탈·아카이브, GPU 텐서 병렬화다. 비2차 맵, 평가 ablation, 규칙 증류도 아직 구현하지 않았다. 솔로 테스트 통과를 전체 게임이나 커리큘럼 ①의 학습 통과로 간주하지 않는다.

```bash
uv run ruff check src tests
uv run ruff format --check src tests
```

## 솔로 PPO

```bash
uv run fog-ridge-train --iterations 20 --episodes-per-update 8 --duration 30 --dimension 2 --output results/ppo
uv run fog-ridge-train --resume results/ppo/checkpoint.pt --iterations 20 --output results/ppo
uv run fog-ridge-train --evaluate results/ppo/checkpoint.pt --eval-episodes 100 --output results/ppo-eval
```

처음에는 SGD 행동 복제로 초기화한다(`--bc-episodes 8 --bc-epochs 20`). `--bc-episodes 0`이면 무작위 초기화에서 시작한다. 기본은 CPU PyTorch이며 작은 직렬 환경에서는 CPU가 적합하다. CUDA 빌드가 설치된 Python 환경에서는 같은 학습 모듈을 `--device cuda`로 실행할 수 있다. 이 인스턴스의 기존 CUDA 환경으로 실행하려면 다음 명령을 쓴다.

```bash
PYTHONPATH=src /venv/main/bin/python -m fog_ridge.ppo --device cuda --output results/ppo-cuda
```

정책은 위치 좌표나 진짜 gap을 입력받지 않는다. 기록장 토큰에 두 층의 set-attention을 적용하고 방향·위치 코사인을 attention bias로 사용한다. 토큰에 슬롯 번호나 위치 인코딩을 넣지 않는다. 점수계 값은 기록장 최솟값에 대한 비율로 정규화한다. 별도 critic만 진짜 gap, 중심까지 거리, 경사 크기와 곡률의 스칼라 통계를 받는다. 두 네트워크 사이에는 공유 파라미터가 없다.

행동 종류, B, 방향 기록 포인터, 목표 위치 포인터, 삭제·요약 슬롯을 범주 분포에서 뽑는다. 요약의 두 번째 포인터는 첫 슬롯을 제외한다. 걸음 세기는 로그 공간의 정규 분포에서 뽑고 디코더에서 지수화한다. 걸음은 선택한 방향과 위치의 조합이며, 현재는 각각 한 기록만 사용하는 one-hot 가중합이다. 원시 샘플을 저장하므로 clipping 후에도 PPO 확률비는 동일한 잠재 행동의 확률로 계산한다. 해당 행동에서 실제 사용하는 헤드만 log probability와 엔트로피에 포함한다.

현재 제한: GRU, 무작위 탐색 벡터, 여러 기록을 동시에 섞는 soft 가중합, 팀원 attention은 없다. 디코더에 탐색 벡터가 없으므로 빈 기록장에서는 걸음을 정책 마스크에서 제외한다. 대기는 1초다. 시간 0인 기록장 결정을 포함해 에피소드 전체를 수집하며 결정 횟수 상한 초과 시 오류로 종료한다.

PPO clip은 0.2, GAE lambda는 0.95, 기본 gamma는 1이다. gamma는 초당 할인으로 `gamma ** dt`를 사용하고 lambda는 결정당 적용한다. 종료 상태 bootstrap은 0이며 에피소드 간 GAE가 섞이지 않는다. advantage 정규화, gradient norm clipping, KL 기준 epoch 조기 종료를 적용한다. 기본 gamma=1일 때 잠재함수 보상 합이 최종 목표와 일치한다. gamma<1 실험은 시간 선호가 달라진다.

출력 파일:

- `checkpoint.pt`: 모델, optimizer, 환경 설정, 학습 반복·다음 시드, 할인·GAE 설정, Torch CPU/CUDA RNG 상태. 재개는 저장된 환경·optimizer·알고리즘 설정을 사용한다.
- `train.jsonl`: 반복별 transition 수, 최종 log gap 중앙값, policy/value loss, entropy, 근사 KL. 재개 시 이어 쓴다.
- `evaluation.json`: 별도 시드 구간에서 결정론적 PPO와 동일한 세계의 SGD 비교 및 행동 횟수.
- `run.json`: 실행 인자와 환경·모델 설정.

평가 시드는 `1_000_000 + seed`부터 사용한다. 현재 SGD는 튜닝 전 기준선이다. 짧은 smoke 학습 성공은 성능 우위를 의미하지 않으며, 충분한 시드와 튜닝된 SGD를 상대로 커리큘럼 통과 여부를 별도로 평가해야 한다.

구현 참고: [PPO 논문](https://arxiv.org/abs/1707.06347), [GAE 논문](https://arxiv.org/abs/1506.02438).

## 교신을 포함한 비동기 MAPPO

```bash
uv run fog-ridge-mappo --players 4 --aggregation min --iterations 20 --output results/mappo-min
uv run fog-ridge-mappo --players 4 --aggregation mean --iterations 20 --output results/mappo-mean
uv run fog-ridge-mappo --resume results/mappo-min/checkpoint.pt --iterations 20 --output results/mappo-min
uv run fog-ridge-mappo --evaluate results/mappo-min/checkpoint.pt --eval-episodes 100 --output results/mappo-eval
uv run fog-ridge-mappo --evaluate results/mappo-min/checkpoint.pt --no-communication --eval-episodes 100 --output results/mappo-no-comm-eval
```

교신은 기본 켜짐이다. `--no-communication`은 학습과 평가 모두에서 교신 행동을 마스킹하는 ablation이다. 위의 마지막 두 명령은 같은 평가 시드로 교신 유무를 비교한다. `--iterations`는 재개 시 추가 학습 횟수다. 체크포인트의 인원, 집계, 환경, 할인·GAE, epoch 수, update당 판 수와 optimizer 상태를 복원한다. `--no-communication`을 명시하면 체크포인트의 교신 설정보다 우선한다. CUDA 환경에서는 `PYTHONPATH=src /venv/main/bin/python -m fog_ridge.mappo --device cuda`로 실행할 수 있다.

### 교신 규칙

- 요청·수락·거절·무시는 즉시 처리하고 응답 없는 요청은 5초 뒤 만료된다. 요청은 상대의 측정·점수계·걸음 중에도 도착해 결정을 깨운다. 무시는 현재 작업을 유지하며, 만료 전에는 이후 결정에서 수락·거절할 수 있다.
- 이미 교신 중이거나 수락 후 걸음을 기다리는 상대에게 요청하면 즉시 403이다. 상대의 busy 상태는 actor 관측이나 요청 대상 마스크로 공개하지 않는다. 수락할 때 두 끝점을 예약하고 경합 요청은 403으로 해제한다.
- 진행 중인 걸음은 끝까지 완료한 후 세션을 시작한다. 양쪽에 걸음이 있으면 모두 완료될 때까지 기다린다. 스냅샷은 이 **실제 세션 시작 시각**의 위치와 마지막 자기 측정 방향을 고정한다. 아직 측정하지 않은 플레이어의 방향은 없다.
- 세션 시작 때 진행 중인 점수계를 취소하고 결과를 폐기한다. 측정은 유지한다. 교신 중 새 측정은 가능하고 걸음·점수계는 불가능하다. 세션 길이는 `max(L_a, L_b)`다.
- 세션 완료 때 고정한 스냅샷을 양쪽 기록장에 넣는다. 세션 중 완료된 측정은 자기 기록장에만 남는다. 스냅샷에는 점수계 값·고도가 포함되지 않는다. 방향과 함께 위치·시각·출처 L/m·측정 B를 보존한다.
- 어느 쪽이든 끊으면 전달 없이 즉시 양쪽을 해제한다. 진행 중 측정은 계속된다. 교신에 의한 자동 이동이나 맵 교체는 없다.
- 동일 시각에는 로컬 행동 완료, 대기 세션 시작, 교신 완료, 요청 만료, 대시보드, T 종료 순서로 처리한다. T에 정확히 끝나는 세션은 전달하며 T를 넘는 세션은 전달하지 않는다. 걸음을 기다리는 예약 단계에서도 끊을 수 있다.

측정 도중 스냅샷을 받더라도 측정 완료 시 기존 기록장을 덮어쓰지 않는다. 같은 시각에 여러 기록이 들어오면 대기 큐에 보존하고 삭제·요약 결정을 강제한다. 로컬 작업은 완료 시각에 그 작업의 변경분만 반영한다. 미래 결과 계산용 복사본의 위치·경사는 actor/critic에 노출하지 않는다.

### 정책과 학습

공유 actor는 자기 기록장과 공개 L/m·대시보드, 자기 요청·교신 상태를 본다. 공개 플레이어 토큰에는 본인 표시와 나에게 온 요청, 내가 보낸 요청, 내 상대 여부가 포함된다. 전원 공통 busy 상태나 절대 플레이어 ID는 주지 않는다. 받은 스냅샷의 방향·위치는 기존 걸음 포인터로 선택할 수 있고, 출처 L/m은 별도 토큰 특징으로 인코딩한다.

행동 헤드는 기존 로컬 행동에 `request`, `accept`, `reject`, `ignore`, `disconnect`, `continue`를 추가한다. 요청·응답은 플레이어 토큰 포인터로 대상을 고르며 해당 행동의 포인터 log probability만 PPO 확률비에 포함한다. `continue`는 작업을 취소하지 않고 다음 사건을 기다리는 제어 응답이다. 알림 도착 시 로컬 작업이 진행 중이면 요청 응답·끊기·강제 기록장 관리만 할 수 있다. 측정이나 이동을 중복 시작할 수 없다.

중앙 critic은 전원의 실제 gap, 중심까지 거리, 경사·곡률 통계, 로컬 작업 남은 시간, focal 플레이어와의 거리, 교신 여부·남은 시간으로 set-attention을 계산한다. actor와 파라미터를 공유하지 않는다.

각 플레이어의 결정부터 그 플레이어의 다음 결정까지 팀 잠재함수 변화와 실제 경과 시간을 누적한다. 요청·응답·교신 종료 알림도 결정 구간을 나눈다. 종료 시 열린 구간을 닫고 에이전트별 GAE를 계산한 뒤 공유 PPO minibatch로 합친다. 기본 `gamma=1`에서 각 플레이어 보상 합은 팀 잠재함수의 전체 변화와 같다.

기본 초기화는 주기적 gossip을 하는 SGD 스크립트의 행동 복제다(`--bc-episodes 4 --bc-epochs 20`). 스크립트는 요청·수락·교신 중 측정을 시연하고, 받은 방향·위치를 얼마나 쓸지는 이후 MAPPO가 학습한다. 스크립트는 상대의 비공개 위치나 방향으로 교신 대상을 선택하지 않는다. `--bc-episodes 0`이면 무작위 초기화부터 시작한다. 평가의 비교 대상은 같은 맵·시작점·L/m·초기 샘플링 시드를 쓰는 독립 SGD 팀이다. MAPPO 평가는 기본적으로 시드별로 재현 가능한 정책 샘플링을 사용하며 학습 RNG를 소비하지 않는다. `--deterministic-evaluation`을 붙이면 argmax로 평가한다. argmax에서는 낮은 확률의 교신 행동이 선택되지 않을 수 있으므로 두 평가 방식을 구분한다.

### 출력과 제한

- `checkpoint.pt`: 모델·optimizer·RNG·환경·학습 설정. 교신 모델은 별도 `communication_version=1`로 저장한다. 이전 교신 없는 MAPPO 체크포인트는 헤드와 관측 크기가 달라 새 모델로 학습을 시작해야 한다. 솔로 PPO 형식은 유지한다.
- `imitation.json`: 모방 데이터 생성 중의 교신 통계.
- `train.jsonl`: 학습 loss·KL·점수와 요청·403·수락·거절·만료·교신 시작/완료·끊기·점수계 취소·스냅샷·중첩 측정 횟수.
- `episodes.jsonl`: 학습 rollout별 시드·플레이어 L/m·gap·결정 횟수·교신 사건 목록. 누가 누구와 언제 교신했는지 학습 중에도 분석할 수 있다.
- `evaluation.json`: 시드별 MAPPO/SGD gap, 교신 통계와 요청·응답·세션 시작/종료 사건 목록. 세션 사건의 끝점으로 교신 그래프를 구성할 수 있다.
- `run.json`: 실행 및 복원된 환경·학습 설정.

아직 이탈·아카이브, 거짓말 변형, GRU, 여러 기록의 soft 가중합, 비2차 맵, GPU 환경 벡터화는 없다. 현재는 인원 고정 배치의 CPU 참조 환경이며 네트워크 자체는 다른 인원과 차원에서도 실행할 수 있다. 짧은 실행으로 교신이 학습 성능을 개선한다고 판단하지 않는다. 요청·403 등은 규칙대로 시간이 0이므로 정책이 이를 끝없이 반복하면 결정 횟수 상한 오류로 중단한다. 별도 cooldown이나 가짜 시간 비용을 추가하지 않는다.

MAPPO 참고: [The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games](https://arxiv.org/abs/2103.01955).
