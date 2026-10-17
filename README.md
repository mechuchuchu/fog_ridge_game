# Fog Ridge

A Python research project that implements [Rulebook v0.5](fog-ridge-rulebook.en.md) and [RL Design v0.2](fog-ridge-rl-design.en.md). It currently supports **solo PPO, asynchronous multi-player MAPPO with communication, dropout, and an archive**, non-quadratic map generation, and paired evaluation.

```bash
uv sync
uv run pytest -q
uv run fog-ridge --episodes 30 --dimension 16 --duration 60 --output results/solo-smoke
```

`results/solo-smoke/summary.json` stores the configuration, the per-seed initial and final gap, the quartiles of the final log10 gap, and action counts. `events.jsonl` stores, at each action completion or at time-out, the elapsed time, whether it completed, the reward, and the true score. This file is a privileged log for evaluation and is not an actor input. Notebook-management actions take zero time and are not included in the current event log.

What is implemented:

- 100 shifted maps, log-uniform diagonal curvature, and closed-form average elevation and gap. The default is quadratic maps; `--curvature-jitter` and `--quartic-strength` turn on per-map curvature differences and a quartic term.
- Continuous-time Measure (`B*m`), Scorer (`3m`), Walk (`m/10`), and Wait.
- Measurement of B distinct maps including the current map. The current map is redrawn only on a completed walk. A zero-distance walk also redraws the map, and the same map can be drawn again.
- Clipping to the maximum walk length, a 5-second dashboard period, and discarding of actions that do not finish by the time limit.
- A finite notebook, delete and direction-record summarize, and forced slot management after a new record is queued when the notebook is full.
- Norm/cosine observations that do not expose coordinate axes, and an action decoder that forms walks as a weighted sum of notebook directions and positions.
- Reward as the difference of the potential `-log10(gap + 1e-12)`, and the actual decision interval `info['dt']`.
- Decaying SGD that picks actions using observations only, periodic Scorer use, and a per-seed evaluation CLI.
- Configurable environment ablations: disabling the Scorer, unit directions, B_max, notebook size, measurement time, latency, maximum stride, and map nonlinearity.

It can also be run directly from Python.

```python
from fog_ridge import Config, SoloEnv
from fog_ridge.baseline import DecayingSGD, rollout

env = SoloEnv(Config(dimension=16, duration=60), seed=42)
result = rollout(env, DecayingSGD(learning_rate=0.08))
print(result)
```

`step()` returns `(observation, reward, done, info)`. An illegal action is rejected with `ValueError` before any state is changed. `info`, `env.landscape`, `env.position`, and `env.records` are simulator/evaluation interfaces. Only the result of `observe()` and the `compose_walk()` decoder are passed to a policy. The PPO network can be used with the training commands below. There is no Gymnasium integration yet.

Simultaneous-time processing follows the order **action completion → dashboard → end**. An action that ends exactly at T completes, and an action that runs past T is discarded. A dashboard update that falls during a measurement uses the previous measured value. The current solo implementation handles dashboard events while skipping over decisions, so a separate heap event queue is not needed.

Notebook summarize is allowed only for two records that have directions. Position and direction are averaged, the time is the latest value, and elevation, score, and B are not merged. Scorer records can be deleted. A notebook that becomes full at the very moment of the end does not affect the final score, so it does not require an extra management decision.

The map noise `spread` uses a per-coordinate standard deviation of `spread / sqrt(D)` so that the typical shift of the center stays similar when the dimension changes. The start position is `start_radius` away from the actual 100-map average center, and the default `S_max` is 1/3 of that distance. The world-generation random numbers and the action-dependent sampling random numbers are separated.

The current baseline performs no learning or tuning. B is fixed, and the learning rate is `lr / (1 + decay * steps)`. It preserves the best Scorer record where possible, but the endgame return is not implemented yet. The Scorer and dashboard values in the observation are also the current raw values, and relative-value normalization must be added at the training stage.

MAPPO can use independent exponentially distributed dropout and a public archive via `--dropout-mean`. On dropout, work in progress is discarded and the last completed position is fixed, and a living policy can choose an archive position as a walk target. The value is a mean in seconds. For example, `--duration 30 --dropout-mean 60` gives a mean dropout time of 2T. The default is no dropout.

Automatic curriculum evaluation, GPU tensor-environment parallelization, full evaluation ablations, and rule distillation are not implemented yet. Passing the solo tests is not to be taken as passing the full game or training for curriculum stage ①.

```bash
uv run ruff check src tests
uv run ruff format --check src tests
```

## Solo PPO

```bash
uv run fog-ridge-train --iterations 20 --episodes-per-update 8 --duration 30 --dimension 2 --output results/ppo
uv run fog-ridge-train --resume results/ppo/checkpoint.pt --iterations 20 --output results/ppo
uv run fog-ridge-train --evaluate results/ppo/checkpoint.pt --eval-episodes 100 --output results/ppo-eval
```

Training is initialized first by SGD behavior cloning (`--bc-episodes 8 --bc-epochs 20`). With `--bc-episodes 0`, it starts from random initialization. The default is CPU PyTorch, and the CPU is appropriate for small serial environments. In a Python environment with a CUDA build installed, the same training module can be run with `--device cuda`. To run with the existing CUDA environment on this instance, use the following command.

```bash
PYTHONPATH=src /venv/main/bin/python -m fog_ridge.ppo --device cuda --output results/ppo-cuda
```

The policy does not take position coordinates or the true gap as input. It applies two layers of set-attention over the notebook tokens and uses the direction/position cosines as an attention bias. No slot numbers or positional encodings are put into the tokens. Scorer values are normalized as a ratio to the notebook minimum. Only a separate critic receives the true gap, the distance to the center, and scalar statistics of gradient magnitude and curvature. No parameters are shared between the two networks.

The action type, B, direction-record pointer, target-position pointer, and delete/summarize slots are sampled from categorical distributions. The walk strength is sampled from a normal distribution in log space and exponentiated in the decoder. A walk is a combination of the selected direction and position, and currently it is a one-hot weighted sum that uses only one record for each. Because the raw samples are stored, the PPO probability ratio is computed as the probability of the same latent action even after clipping. Only the heads actually used by that action are included in the log probability and entropy.

Current limitations: there is no GRU, no random exploration vectors, no soft weighted sum that mixes several records at once, and no teammate attention. Because the decoder has no exploration vector, walks are excluded by the policy mask when the notebook is empty. Wait is 1 second. The whole episode is collected, including zero-time notebook decisions, and exceeding the decision-count limit terminates with an error.

The PPO clip is 0.2, GAE lambda is 0.95, and the default gamma is 1. Gamma is a per-second discount using `gamma ** dt`, and lambda is applied per decision. The bootstrap at terminal states is 0, and GAE is not mixed across episodes. Advantage normalization, gradient norm clipping, and early epoch termination based on KL are applied. With the default gamma=1, the sum of the potential-function rewards matches the final objective. Experiments with gamma<1 change the time preference.

Output files:

- `checkpoint.pt`: model, optimizer, environment configuration, training iteration and next seed, discount/GAE settings, and Torch CPU/CUDA RNG state. Resuming uses the saved environment, optimizer, and algorithm settings.
- `train.jsonl`: per-iteration transition count, median final log gap, policy/value loss, entropy, and approximate KL. Appended to on resume.
- `evaluation.json`: deterministic PPO versus SGD on identical worlds over a separate seed range, plus action counts.
- `run.json`: run arguments and the environment and model configuration.

Evaluation seeds start from `1_000_000 + seed`. The current SGD is an untuned reference baseline. Success in a short smoke training run does not imply a performance advantage, and whether the curriculum stage is passed must be evaluated separately against a tuned SGD with enough seeds.

Implementation references: [PPO paper](https://arxiv.org/abs/1707.06347), [GAE paper](https://arxiv.org/abs/1506.02438).

## Asynchronous MAPPO with Communication

```bash
uv run fog-ridge-mappo --players 4 --aggregation min --iterations 20 --output results/mappo-min
uv run fog-ridge-mappo --players 4 --aggregation mean --iterations 20 --output results/mappo-mean
uv run fog-ridge-mappo --players 4 --aggregation min --iterations 20 --duration 30 --dropout-mean 120 --output results/mappo-dropout
uv run fog-ridge-mappo --players 4 --aggregation min --iterations 20 --curvature-jitter 0.1 --quartic-strength 0.2 --output results/mappo-nonlinear
uv run fog-ridge-mappo --players 4 --aggregation min --iterations 20 --no-score --unit-directions --batch-max 1 --output results/mappo-ablation
uv run fog-ridge-mappo --players 4 --aggregation min --iterations 20 --dropout-mean 120 --hide-archives --output results/mappo-hidden-archives
uv run fog-ridge-mappo --players 4 --aggregation min --iterations 20 --homogeneous --latency 2 --measurement-time 0.5 --output results/mappo-slow-link
uv run fog-ridge-mappo --resume results/mappo-min/checkpoint.pt --iterations 20 --output results/mappo-min
uv run fog-ridge-mappo --evaluate results/mappo-min/checkpoint.pt --eval-episodes 100 --output results/mappo-eval
uv run fog-ridge-mappo --evaluate results/mappo-min/checkpoint.pt --no-communication --eval-episodes 100 --output results/mappo-no-comm-eval
```

Communication is on by default. `--no-communication` is an ablation that masks communication actions in both training and evaluation. The last two commands above compare with and without communication on the same evaluation seeds. `--iterations` is the number of additional training iterations when resuming. The checkpoint's number of players, aggregation, environment, discount/GAE, number of epochs, number of games per update, and optimizer state are restored. If `--no-communication` is given explicitly, it takes precedence over the checkpoint's communication setting. In a CUDA environment it can be run with `PYTHONPATH=src /venv/main/bin/python -m fog_ridge.mappo --device cuda`.

### Communication Rules

- Request, accept, reject, and ignore are handled immediately, and an unanswered request expires after 5 seconds. A request arrives even while the partner is measuring, using the Scorer, or walking, and wakes up a decision. Ignore keeps the current work, and before expiry the request can still be accepted or rejected at a later decision.
- A request to a partner who is already communicating or who is waiting for a walk after accepting returns an immediate 403. The partner's busy state is not exposed through the actor observation or the request-target mask. At acceptance, both endpoints are reserved, and competing requests are released with a 403.
- A walk in progress is completed to the end before the session starts. If both sides have walks, the session waits until both are done. The snapshot fixes the position and the last self-measured direction at this **actual session start time**. A player who has not measured yet has no direction.
- At session start, a Scorer in progress is cancelled and its result is discarded. Measurement is kept. A new measurement during communication is allowed, and walking and the Scorer are not. The session length is `max(L_a, L_b)`.
- At session completion, the fixed snapshot is put into both notebooks. A measurement completed during the session stays only in its own notebook. The snapshot does not include the Scorer value or elevation. Along with the direction, it preserves the position, time, source L/m, and measurement B.
- If either side disconnects, both are released immediately with no delivery. A measurement in progress continues. There is no automatic movement or map replacement caused by communication.
- At the same time instant, events are processed in the order: local action completion, pending session start, communication completion, request expiry, dashboard, T end. A session that ends exactly at T is delivered, and a session that runs past T is not. A player can also disconnect during the reserved stage while waiting for a walk.

Even if a snapshot is received during a measurement, the existing notebook is not overwritten when the measurement completes. If several records arrive at the same time, they are kept in a pending queue and a delete/summarize decision is forced. A local action applies only that action's own changes at its completion time. The position and gradient of the copy used to compute future results are not exposed to the actor/critic.

### Policy and Training

The shared actor sees its own notebook, the public L/m and dashboard, and its own request and communication state. The public player token includes a self marker, requests addressed to me, requests I sent, and whether the player is my partner. Neither a busy state common to everyone nor an absolute player ID is provided. The direction and position of a received snapshot can be selected with the existing walk pointers, and the source L/m is encoded as a separate token feature.

The action head adds `request`, `accept`, `reject`, `ignore`, `disconnect`, and `continue` to the existing local actions. Request and response choose their target with a pointer over player tokens, and only that action's pointer log probability is included in the PPO probability ratio. `continue` is a control response that waits for the next event without cancelling the work. When a notification arrives while local work is in progress, only responding to a request, disconnecting, and forced notebook management are possible. A measurement or movement cannot be started twice.

The central critic computes set-attention over every player's true gap, distance to the center, gradient/curvature statistics, remaining time of local work, distance from the focal player, and whether communicating and the remaining time. It shares no parameters with the actor.

From each player's decision to that player's next decision, the change in the team potential function and the actual elapsed time are accumulated. Request, response, and communication-end notifications also split the decision intervals. At the end, open intervals are closed, per-agent GAE is computed, and the result is merged into shared PPO minibatches. With the default `gamma=1`, the sum of each player's rewards equals the total change in the team potential function.

The default initialization is behavior cloning of an SGD script that does periodic gossip (`--bc-episodes 4 --bc-epochs 20`). The script demonstrates requesting, accepting, and measuring during communication, and how much of the received direction and position to use is learned later by MAPPO. The script does not choose communication partners using a partner's private position or direction. With `--bc-episodes 0`, it starts from random initialization. The evaluation comparison target is an independent SGD team that uses the same maps, start points, L/m, and initial sampling seeds. MAPPO evaluation by default uses policy sampling that is reproducible per seed and does not consume the training RNG. Adding `--deterministic-evaluation` evaluates with argmax. Under argmax, low-probability communication actions may not be selected, so the two evaluation modes should be distinguished.

### Output and Limitations

- `checkpoint.pt`: model, optimizer, RNG, environment, and training configuration. The communication/archive model is saved with a separate `communication_version=2`. Earlier MAPPO checkpoints have a different observation size, so training must start with a new model. The solo PPO format is unchanged.
- `imitation.json`: communication statistics during imitation-data generation.
- `train.jsonl`: training loss, KL, and score, plus counts of requests, 403s, accepts, rejects, expiries, communication starts/completions, disconnects, Scorer cancellations, snapshots, and overlapped measurements.
- `episodes.jsonl`: per-training-rollout seed, player L/m, gap, decision count, and list of communication events. It lets you analyze who communicated with whom and when, even during training.
- `evaluation.json`: per-seed MAPPO/SGD gap, the paired median log10-gap difference and its 95% bootstrap interval, communication statistics, and the lists of request, response, and session events. A communication graph can be built from the session endpoints.
- `run.json`: the run and the restored environment and training configuration.

Still missing are the lying variant, GRU, soft weighted sums of several records, GPU environment vectorization, curriculum automation, automatic rule distillation, and a runner for the full set of ablations. The default environment configuration can be changed from the CLI to train a single ablation, and `--no-communication` supports same-seed evaluation comparison. The current environment is a CPU reference environment with a fixed number of players per batch, and the network itself can run with a different number of players and dimensions. Do not conclude from short runs that communication improves learned performance. Because requests, 403s, and the like take zero time by the rules, if the policy repeats them endlessly it aborts with the decision-count limit error. No separate cooldown or artificial time cost is added.

MAPPO reference: [The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games](https://arxiv.org/abs/2103.01955).
