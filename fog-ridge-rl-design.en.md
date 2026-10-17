# Fog Ridge — RL Design (Big Picture) v0.2

Reference rulebook: v0.5 (time limit T, team score = the true score of the lowest player, random dropout, public archive)
Purpose: **to obtain hints for distributed-training protocols** (Section 1)

---

## 0. At a Glance

**What we build**: **one shared policy** that plays this game well. Everyone uses the same network, and the only differences between players are L, m, position, and notebook.
The policy learns "when to measure, how much to gather, whom to communicate with, and in which direction and how far to walk." In effect it is a **learned optimizer + communication protocol**.

| Item | Choice |
|---|---|
| Simulator | Maps are built as quadratic functions so elevation and gradient are computed in closed form. Thousands of games run in parallel on GPU |
| Time | Continuous time driven by an event queue. A semi-MDP in which each agent has different decision times |
| Observation | Public information + my notebook. Vectors are not seen directly; only **rotation-invariant features** such as inner products and norms are |
| Action | A discrete action type (masked) + parameters. The walk direction is output as a **weighted sum of notebook vectors** |
| Reward | Shaping that splits −log(final gap) across time. Privileged information is used only during training |
| Algorithm | MAPPO (parameter sharing, central critic), starting from imitation of a scripted policy |
| Curriculum | Solo → 2 players → many players → dropout → high dimension → (optional) lying |
| Evaluation | Paired comparison on the same seeds, per-mechanism ablations, minimum and mean aggregations run in parallel. The "independent solo N players" baseline is the key |
| Deliverable | Behavior analysis → rule distillation → simulator verification → real-world translation |

## 1. Purpose: Hints for Distributed-Training Protocols

The goal of this RL is not the policy that plays the game well in itself, but to extract **rules usable in real distributed training** from that policy's behavior. The results go through the following sequence.

1. Look at the trained policy's behavior as statistics (behavior statistics in Section 10).
2. **Distill the recurring patterns into simple rules** (a decision tree or a few lines of heuristics).
3. Pit the distilled rules against the policy and the baselines in the simulator to check **whether the rules alone achieve the performance**.
4. Convert into real-world terms using the "Translation" column of Rulebook Section 11. Behaviors that arise from **fun-only** rules are excluded from the hints.

Questions to answer:
- When to start communicating (what is the signal that progress has stalled)?
- With whom (which matters: L, m, distance, or snapshot age)?
- How much of what is received to mix in (how does the ratio shrink as the snapshot gets older or farther)?
- When and how to change B and the stride (relation to the signal-to-noise ratio)?
- How often to verify (Scorer), and how to leave checkpoints?
- How does strategy change as dropout gets more frequent?
- When does role differentiation (leader/laggard) pay off?

Effects on the design:

| Item | Policy |
|---|---|
| Simulator | Includes non-quadratic maps and per-map curvature differences as **mandatory** (Section 3.4). A rule that works only on quadratics is not a hint |
| Score aggregation | Train and evaluate both the minimum (best-of-N) and the **mean (consensus model)**. Real training usually aims at a single consensus model, so behaviors that appear under both aggregations are more credible hints |
| Ablation | Change one mechanism at a time and see how the policy changes (L distribution, B_max, S_max, dropout rate, notebook size, presence of the archive) |
| Teammates | A policy where everyone is the same is enough. Compatibility with humans is not needed. However, mix in some past checkpoints so the policy does not overfit to conventions only it knows |
| Logs | Save all behavior statistics (Section 10) |

## 2. Confirmed Rule Interpretations (Rulebook v0.5)

The simulator implements the following as defaults, and each can be changed with a flag (for ablation).

| # | Item | Confirmed content |
|---|---|---|
| 1 | Maps a measurement draws | Current map + B−1 random maps drawn fresh each measurement. Re-measuring in place with B=1 gives the same value |
| 2 | Maximum distance of one walk S_max | 1/3 of the distance R0 from the start to the average bottom. Given to the policy as a scalar (randomized per game) |
| 3 | Magnitude of the descent direction | Included (the gradient vector as is). Unit-vector mode is for ablation |
| 4 | Request timing | Requests and responses are always immediate. At session start the Scorer is cancelled, Measure continues, and a walk finishes before the session starts. Request expiry is 5 seconds |
| 5 | Start position | Default is the same point for everyone. Only during training, some scattered starts are mixed in; evaluation uses the same point |
| 6 | Scorer time | s = 3m (per player) |

## 3. Simulator

### 3.1 Map Model

- Position x ∈ ℝᴰ, elevation of map i: f_i(x) = ½ (x − c_i)ᵀ A (x − c_i). The bottom position c_i differs per map (c_i = center + σ_c·noise).
- A = diag(a_1 … a_D), where a_j is log-uniform on [1, κ] (κ is the condition number). The policy is designed not to see the coordinate axes (Section 4), so even though A is diagonal, axis alignment cannot be exploited.
- Let c̄ = the mean of the 100 c_i. Then:
  - **100-map average elevation** f̄(x) = ½ (x − c̄)ᵀ A (x − c̄) + F*, where F* = mean of ½ (c_i − c̄)ᵀ A (c_i − c̄) is the reachable floor.
  - **gap(x) = f̄(x) − F***. The internal score uses the log of this gap.
- When measuring with a set of maps S (size B, including the current map):
  - Elevation = the mean of f_i(x) over S.
  - Descent direction = −A(x − c̄_S) = **true descent direction** −A(x − c̄) + **noise** −A(c̄ − c̄_S).
- The noise magnitude is roughly σ_c/√B regardless of x, while the true gradient goes to 0 as you approach the bottom. So **the noise growth near the bottom (Rulebook Section 8) arises on its own without separate implementation.**
- Team score = min_j gap(x_j) (all living players and archive positions). The simulator knows the exact value.

### 3.2 Time and Events

- Events: action completion, dashboard update (5 s), dropout, T, request arrival and expiry, communication end.
- Agent state: the action in progress and its end time (busy_until), the communication partner, pending requests, and whether alive.
- **Decision points**: when an action ends, or when a waking event occurs. Waking events are **request arrival** (whatever action the recipient is doing), communication end, and partner dropout.
  - When a request arrives, the recipient's action in progress is left as is, and one additional decision point is given to choose among {accept, reject, ignore}. If ignored, the request expires after 5 seconds.
- The effect of an action is applied at its **completion time**. A dashboard update reflects only the last measurement that had **finished** by that time.
- **Cancellation rule at session start**: the Scorer is cancelled (no result), Measure continues (result goes to my notebook), and a walk finishes before the session starts. Since the Scorer is the only action that can be cancelled, handling preemption is simple.
- One micro-step of a game = one agent that hit its next decision point choosing an action. Even if the number of micro-steps differs per game, they can be batched together.
- Semi-MDP: rewards are summed over each agent's decision interval Δt and discounted by γ^Δt. Since the reward is effectively concentrated at the end, γ is set close to 1 per second.

### 3.3 Vectorization and Memory

- State is held as [game, player, …] tensors, and event handling is done with masks.
- Memory is dominated by the notebook [games E, players P, slots N, dimension D]. If both position and direction are stored, with E=4096, P=8, N=20 it is about 0.7 GB at D=128 and about 2.7 GB at D=512 (float32). In high dimensions, reduce the number of games.

### 3.4 Moving Beyond Simplicity

A quadratic function has too clean an answer structure, so the policy can exploit it (for example, estimating A from the gradient difference between two points and taking Newton steps). This is a legitimate strategy in itself, but it does not guarantee generalization. Since the goal is hints, this stage is not optional but mandatory.
- Make A_i slightly different per map (A_i = A + a small perturbation).
- Add non-quadratic terms (quartic terms, curved valleys like the Rosenbrock function).
- Mix in non-quadratic maps from mid-training, and **always use non-quadratic maps for evaluation and for verifying distilled rules.** Rules that hold only for quadratics are excluded from the hints.

## 4. Observation

All vectors are handled as **vectors relative to my current position**, and the policy is given not the vectors themselves but only norms, inner products, and cosines. That way the same policy runs across different dimensions and cannot see the coordinate axes.

### Public Information (everyone sees the same thing)

- Remaining time / T, dropout probability distribution, S_max.
- Per-player token: L, m, t_s, s, latest dashboard elevation and its age, the change over the last few updates, dropout status.
- Per-archive-entry token: time elapsed since dropout, last measured elevation, distance from my position.
- Elevations are given not as absolute values but as relative values such as "ratio to the dashboard minimum" (since scale differs per game).

### Private Information (only I see it)

- Per-notebook-entry token (up to N): kind (my measurement / received snapshot / Scorer / place visited from the archive), age, B, the source's L and m, direction norm, position distance, cosine between the direction and the position direction, Scorer value (relative to the best Scorer value).
- **Pairwise bias**: cosines between the directions of entries i and j, and between the direction of i and the position direction of j. Agreement between entries (whether the samples point in the same direction) is the measure of noise.
- R random exploration vectors (random directions generated by the simulator each time) as tokens. Needed to explore outside the notebook.
- Current state: the action in progress and its remaining time, communication partner and elapsed time, pending requests (sent / received), last Scorer value and its age, number of empty slots.

### Privileged Information (only the critic sees it)

- Every player's true position (relative to c̄) and true gap, the current leader, c̄, A, and dropout status.

## 5. Actions

| Action | Parameters | Legal condition (mask) |
|---|---|---|
| Measure | B ∈ {1 … B_max} | Always possible (also during communication) |
| Scorer | — | When not communicating |
| Walk | Direction weights π, strength η, length limit S_max | When not communicating |
| Request | Partner j (pointer) | j is alive and I have no request outstanding |
| Accept / Reject | — | When there is a received request (immediately, even while I am doing another action) |
| Ignore | — | Only at the decision point where the request arrived. Continue the action in progress; the request expires after 5 seconds |
| Disconnect | — | When communicating |
| Summarize / Delete | Notebook slot (pointer) | When there are two or more records. **A forced decision when the notebook is full and a new record arrives** |
| Wait | Until the next event, or a fixed time | Always |

At the moment of accepting, a Scorer in progress is cancelled and 3m is lost. So the policy must weigh "whether to start the Scorer before a request arrives" and "whether to wait for a request that arrived during the Scorer to finish (5-second expiry)." A requester also faces the same risk if it starts the Scorer while waiting for a response.

**Walk direction output**: rather than emitting a D-dimensional vector directly, it is built from a combination of notebook vectors.

  walk = η_g · Σ π_k g_k  +  η_p · Σ π_k (p_k − x)  +  η_r · Σ π_k r_k

- g_k: notebook directions (corresponds to averaging, momentum, and weighting partner samples).
- p_k − x: pulling toward notebook positions (corresponds to the merge ratio after communication. If π concentrates on one place, you go to that position; if it is split, you go to the midpoint).
- r_k: random exploration vectors.
- π is the softmax of attention logits derived from the entry tokens, and η is a log-scale Gaussian. The total walk length is clipped to S_max.

This way the same policy runs across different dimensions and is human-readable ("average the 3 most recent samples and move 20% toward the partner").
Calling the Scorer right after a walk acts as a trial-walk effect (before/after comparison), and the policy must learn this judgment.

## 6. Reward

### Main Reward

- gap_t = min_j gap(x_j) (living players + archive). Φ_t = −log₁₀(gap_t + ε).
- At each event, the team reward is Δ = Φ(next) − Φ(current). **Summed, this is Φ_T − Φ_0, which equals the final objective (−log final gap).** That is, it is potential-function shaping, so it does not change the optimal policy while distributing the signal over time.
- In the semi-MDP, each agent is given **the sum of the team reward over its own decision interval.**
- Reason for using the log: a 100× reduction in gap early and a 2× reduction late can be handled together in "orders of magnitude," and the scale becomes similar even when dimension and condition number differ.
- The gap is known only to the simulator (privileged). The policy cannot see the gap directly.
- In the mean-aggregation experiment, the same kind of Φ is used with the **mean gap** (dropped players at their archive positions) instead of min_j. In that case laggards also receive a direct signal, so the auxiliary reward below is not needed.

### Auxiliary Reward (to supplement credit assignment)

- Under the minimum score, only the leader's actions have marginal contribution, and laggards get almost no signal.
- Mix in the log increment of the mean gap with weight λ (e.g., 0.3), and decay it to 0 as training progresses. Since this is a bias that changes the optimal policy, **evaluate with it turned off**, and compare on versus off as an ablation.
- Alternatives: a counterfactual baseline (COMA-style) or a difference reward. The simulator is cheap so these are candidates, but they are heavy to implement.

### Dropped-out Agents

- A dropped agent's trajectory ends there (loss mask), and there is no penalty for the dropout event.
- The position just before dropout becomes a team-score candidate, so "being somewhere it is fine to leave from at any time" emerges naturally from the reward structure.

## 7. Policy and Critic

- **Encoder**: 2–3 layers of set-attention over item tokens (notebook + archive + random vectors) with pairwise bias, a separate attention for player tokens, and an MLP for scalar context. Independent of order and slot numbers.
- **Temporal memory**: one short GRU layer (for progress state such as "a request I sent is still unanswered"). Most information is already contained in the observation.
- **Heads**: action type (softmax with mask), B (categorical), partner choice (pointer over player tokens), walk weights π (per-item attention), strength η (log-Gaussian), summarize/delete slot (pointer).
- **Critic**: a central critic using privileged information. It takes the player-set tokens + time + dropout state + agent i's context as input and outputs a per-agent value V(s, i).
- **Roles come from state, not IDs**: agent numbers are not given. Roles such as leader/laggard and hub must be determined from (L, m, position rank, notebook), so that the policy generalizes when the number of players changes.

## 8. Training

### Algorithm

- **MAPPO**: parameter sharing, central critic, GAE, PPO clip. Entropy is weighted most heavily on the action-type head and gradually reduced.
- Games vary in length, so samples are stacked in micro-step units and reconstructed into per-agent trajectories.
- Start from standard PPO defaults (clip 0.2, GAE λ near 0.95), but since the minimum objective has high variance per game, use a **large batch**.

### Domain Randomization

| Element | Range (placeholder) |
|---|---|
| L | Log-uniform 0.25 – 5 s. Some games have all players identical, some have an extreme mix |
| m | Log-uniform 0.5 – 2 s |
| Players | 2 – 12 |
| Dimension D | 2, 8, 16, 64, 128, 512 (depending on the curriculum stage) |
| Condition number κ | 1 – 100 |
| Map-to-map difference σ_c | 5% – 50% of the start radius |
| T | 0.5 – 1.5× the rulebook value (the policy sees the remaining time) |
| Dropout | No dropout / mean 4T / 2T / T |
| S_max, start position | Items in Section 2 |

### The Exploration Problem and the Starting Point

- Communication has its cost immediately and its benefit much later. Using PPO alone from the start risks converging to never communicating.
- **Start from a scripted policy**: build solo SGD (with decay) + periodic averaging with a nearby partner, etc., initialize with behavior cloning, and improve with PPO while gradually reducing the KL to the script.
- **Start with a small L**: let the policy first experience the usefulness of communication in an environment where it is cheap, then generalize by raising L.
- **Diversify teammates**: if only identical policies play together, conventions (implicit agreements) known only to that policy arise, and such conventions cannot be carried over into hints. Mix in past checkpoints and scripted bots as teammates at some ratio, so that only behaviors that work with any partner remain.

## 9. Curriculum

| Stage | Setup | Pass criterion |
|---|---|---|
| ① Solo | 1 player, D 2 → 16, no dropout | At or above the median of tuned solo SGD |
| ② 2 players | D=16, heterogeneous L/m | Significantly better than a policy that does not communicate |
| ③ Many players | 4 – 8 players, 403, mutual acceptance, no dropout. **Train the minimum and mean objectives separately** | Better than N independent solos (for the minimum, against the best of them; for the mean, against the average) |
| ④ Dropout | Mean dropout 4T → 2T → T, archive | Performance degrades gracefully as the dropout rate rises |
| ⑤ Scale | D 128 → 512, non-quadratic maps, A_i variants | Holds at D and player counts not seen in training |
| ⑥ (Optional) | Lying variant, saboteur | Verification behaviors that avoid being deceived (Scorer, walking there to check) emerge |

## 10. Evaluation

### Baselines

1. **Best of N independent solos**: each player descends alone with tuned SGD, and the player who ends best is used. Under the minimum score, the diversity of independent attempts comes for free, so this is **the most important comparison target**.
2. Periodic global averaging (idealized all-reduce).
3. Random-pair gossip averaging.
4. Local SGD (merging at fixed intervals).
5. Leader-following heuristic: a rule that follows the dashboard minimum.
6. Oracles: an optimizer that knows the true gradient (upper bound on the optimization side), and a team with free communication (upper bound on the protocol side).

### Metrics

- Median and interquartile range of the final log gap (at time T).
- Score curves over time, and the time to first reach a target gap.
- Changes when varying dropout rate, L/m distribution, number of players, and dimension (including values not seen in training).
- Behavior statistics: time allocation (Measure / Scorer / communication / waiting), time evolution of B and stride, communication graph (who with whom), 403 ratio, stability of the leader, distance between laggards and the leader, frequency of picking up from the archive, and whether the policy returns to a checkpoint in the endgame.

### Methods

- **Paired comparison on the same seeds**: apply the same maps, the same start positions, and the same dropout times to all policies. Minimum statistics are noisy, so run many games (thousands) and report bootstrap confidence intervals.
- Ablation (functions): no communication / no Scorer / B=1 fixed / hidden archive / no dropout / unit-vector direction.
- Ablation (mechanism values): vary the L distribution, B_max, S_max, notebook size, and dropout rate and see how the policy changes. **Which behaviors appear under which conditions** is the raw material for the hints.
- **Two aggregations**: train separately on the minimum and mean score objectives and compare. Behaviors that appear under both (B and stride schedules, communicating when stalled, etc.) are credible hints, while behaviors that appear only under the minimum (concentrating on the leader, successors) are interpreted only for best-of-N selection settings.
- **Basic check**: at N=1 it must be at or above tuned solo SGD. If this fails, nothing else is meaningful.

### Hint Extraction

- Organize behavior statistics as functions of conditions (L, m, noise ratio, snapshot age, remaining time).
- Distill rules with decision trees or short formulas, and compare a team using only those rules against the policy and the baselines in the simulator. If it recovers most of the policy's performance, adopt the rule as a hint.
- For each adopted rule, check the real-world correspondence using the translation column of Rulebook Section 11, and **discard rules that depend on fun-only elements.**
- Check that the rule also holds on non-quadratic maps and at D, player counts, and dropout rates not seen in training.

## 11. Expected Behaviors (Hypotheses)

Existing hypotheses:
- Early: B=1 with large strides; late: B↑ and stride↓.
- Use the Scorer sparingly, and cluster it around trial walks.
- Low-latency players pair up, and slow players buy time by measuring during communication.
- Communicate more when improvement stalls.

New expectations from the minimum score and dropout:
- **Leader/laggard differentiation**: one player descends while the rest supply samples near it.
- **Asymmetric merging**: laggards walk toward the leader, and the leader barely moves.
- **Exploration role**: some players try other valleys to create diversity.
- **Successor**: insurance by placing a second player near the leader.
- **Picking up from the archive**: go to a dropped player's last position, check it with the Scorer, and continue descending from there.
- **Endgame return**: return to the Scorer-verified best point to meet T, and time things so that no action in progress runs past T.

These hypotheses remain hypotheses until confirmed. If they do not appear, suspect the reward structure or the exploration problem first.

## 12. Pitfalls

- **Simulator exploitation**: using measurement time during communication, exploiting map replacement after a walk, abusing the request expiry time, re-measuring the same map, etc. If you see behavior that deviates from the rule interpretations (Section 2), suspect the simulator before the policy.
- **Quadratic bias**: the policy may learn shortcuts fitted to quadratic structure. Always test with non-quadratic maps.
- **Multi-agent non-stationarity**: teammates change together, so training wobbles. Counter with teammate diversification, large batches, and low learning rates.
- **Credit assignment**: under the minimum score, a laggard's contribution is invisible. Check by ablation whether an auxiliary reward or a counterfactual baseline is needed.
- **High-variance objective**: the outcome of a single game depends heavily on luck. Reduce this with the critic's privileged information and large batches.
- **A local optimum of never communicating**: the imitation initialization and the small-L start in Section 8 are the defenses.
- **Overfitting to dropout timing**: randomize the dropout-time distribution widely.
- **Mistaking fun-only rules for real hints**: behaviors from rules with no real counterpart, such as private Scorer results or discrete map replacement, are not hints. Check the translation column for every distilled rule.
- **Difference between a consensus model and best-of-N**: strategies that appear only under the minimum (concentrating on the leader) must not be carried over to real training as is. Cross-check against the mean-objective results.

## 13. Roadmap

0. (Done) Fix the purpose and rule interpretations. Freeze the simulator rules according to Rulebook v0.5.
1. Solo simulator + scripted baseline. Unit tests:
   - The closed-form elevation/gradient matches the value obtained by directly averaging the 100 maps.
   - Event ordering and application at completion time are correct.
   - Out-of-mask actions are never executed.
   - The same seed gives exactly the same result.
   - Invariants are never broken even when running at scale with a random policy.
2. Solo training with the policy and the PPO skeleton (pass stage ①).
3. Many players + communication + 403, with both the minimum and mean objectives (stages ②③).
4. Dropout + archive (stage ④).
5. Scale expansion, non-quadratic maps, full evaluation and ablations (stage ⑤).
6. Result analysis: behavior statistics → rule distillation → simulator verification → real-world translation (hint extraction in Section 10). If signs appear that a rule depends on a dominant strategy or fun-only elements, feed back to the rulebook.
