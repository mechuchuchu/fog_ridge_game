# 안개 능선

[룰북 v0.5](fog-ridge-rulebook.md)와 [RL 설계 v0.2](fog-ridge-rl-design.md)를 구현하는 Python 연구 프로젝트다. 현재는 **솔로 참조 시뮬레이터, 스크립트 베이스라인과 PPO 학습**을 구현했다. 다인 교신과 MAPPO는 후속 단계다.

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

다음 구현 단계는 솔로 정책 튜닝과 커리큘럼 ① 평가, 이후 다인 이벤트 큐와 교신(요청/수락/403), 이탈·아카이브, GPU 텐서 병렬화다. 비2차 맵, MAPPO, 평가 ablation, 규칙 증류도 아직 구현하지 않았다. 솔로 테스트 통과를 전체 게임이나 커리큘럼 ①의 학습 통과로 간주하지 않는다.

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
