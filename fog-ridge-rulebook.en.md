# Fog Ridge (working title) — Rulebook v0.5

A **real-time cooperative game** in which several players descend a foggy mountain. Each player can only see what is at their feet; they communicate to pool information and try to push the team's **deepest point** as low as possible before the time limit.
If the number of position dimensions is raised to several hundred, the game becomes a toy version of "asynchronous gossip SGD, where every node has a different speed and latency and nodes drop out midway."

**Purpose**: to translate the behaviors that RL agents discover in this game into **hints for distributed-training protocols**. For that reason, the rules are designed, where possible, to correspond to elements of real distributed training (batch size, latency, stale information, dropout, checkpoints). Rules with no such correspondence are marked as fun-only (Section 11).

---

## 1. Goal and Score

- **End**: the game ends when the time limit **T** has elapsed (placeholder example: 600 seconds). The remaining time is visible to everyone.
- **Team score** = the **lowest value, at time T, of every player's "100-map average elevation at their current position."** **Lower is better.**
  - Players who dropped out are also candidates. A dropped-out player is evaluated at the **position left in the archive**.
  - The score is computed from the **true 100-map average elevation**, not from the measured elevation shown on the dashboard. A measurement that happened to come out low by luck does not count as score.
  - An action in progress at time T is void. If a walk has not finished, the previous position is used.
- A player can only learn the true score of their own position by using the **Scorer**.
- Players: **6** (placeholder example; experiments vary this from 2 to 12).

### Dropout and the Archive

- A player may **drop out at a random time** during the game. Dropout happens without warning, and **there is no re-entry.**
- Dropout is an independent random event for each player. The distribution of dropout probability is public to everyone, but nobody knows any individual's dropout time.
  - Placeholder example: an exponential distribution with mean dropout time 2T (about 40% of players drop out per game).
- At the moment of dropout, the following is recorded in the **archive** (public to everyone):
  - Dropout time, position at dropout, last measured elevation, and that player's L and m.
  - The notebook, Scorer results, and descent direction are not kept.
- Dropout handling:
  - The action in progress (Measure, Scorer, or Walk) is discarded, and the position is that of the last completed walk.
  - If the player was communicating, the partner is released immediately. Nothing is delivered (same as a disconnect).
  - Requests the player sent or received become void, and any later request to them gets a 403.
  - Their row on the dashboard is frozen at its last value and marked "dropped out."
- Anyone can walk to a position in the archive. Only the position is public; there is no direction.

## 2. The World

- **100** maps. All are variants of the same mountain, and **each map's bottom is at a slightly different position.**
- At the start, **everyone departs from the same point**. Maps are randomly assigned to each player (duplicates allowed).
- Each player has a "**current map**." The current map is **replaced with a random one of the 100 each time a walk ends.** It never changes except through walking.
- **Continuous time**, no turns.
- Positions start in 2D and can be extended to several hundred dimensions.

## 3. Values

### Per-player fixed values (never change during the game; public to everyone)

| Value | Meaning | Example (placeholder) |
|---|---|---|
| L | Communication latency | 0.25 – 5 s |
| m | Measurement time **per map** | about 1 s |

### Derived values (differ per player; public to everyone)

| Value | Meaning |
|---|---|
| t_s = **m / 10** | Time for one walk. Independent of walk distance |
| s = **3m** | Time to use the Scorer |

### Common values (placeholder examples)

| Value | Meaning | Example |
|---|---|---|
| T | Time limit | 600 s |
| S_max | Maximum distance of one walk | 1/3 of the distance from the start to the average bottom (published as a number) |
| B_max | Maximum number of maps that can be overlaid in one measurement | 8 |
| N | Number of notebook slots | 20 |
| Dashboard update period | Interval at which elevations are published | 5 s |
| Request expiry time | Time after which an unanswered communication request becomes void | 5 s |

## 4. Actions

A player can do only one thing at a time. Exception: Measure is allowed during communication.

| Action | Time taken | What you get / effect |
|---|---|---|
| **Measure** (B maps) | B × m | The **average elevation** over the current map plus B−1 random maps, overlaid, plus the **average descent direction** (gradient vector, magnitude included; private, stored in the notebook). The elevation is published at the next dashboard update |
| **Scorer** | s (= 3m) | The **100-map average elevation** (the true score) at my position. Does not reveal direction. Stored in my notebook only |
| **Walk** | t_s (= m/10) | In the desired direction, up to S_max. The current map is replaced when it ends. How far to go is each player's choice |
| **Communicate** | max(L_a, L_b) | Exchange snapshots of each other's (position, descent direction) |
| **Summarize** | Instant | Merge two notebook records into one |

### Batch Size and Maps

- When measuring, you choose B maps (1 to B_max). **My current map is always included**, and the remaining B−1 are drawn fresh at random for each measurement.
- Therefore **measuring again on the spot with B=1 gives the same value.** To get a new sample, you must walk (which replaces the current map) or increase B.
- The error shrinks by roughly **1/√B**.
- Time grows B-fold while error shrinks only √B-fold, so the larger B is, the worse the efficiency.

## 5. Communication Rules

1. **Only one communication at a time.** If the other player is already communicating, the request immediately returns a **403 (refusal)**, with no time lost.
2. It starts by **mutual acceptance**. Requests and responses (accept / reject) are **always delivered immediately, whatever the other player is doing.** If no response comes, the request **expires after 5 seconds**. Before acceptance, both the requester and the recipient can move freely.
3. Communication time is **max(L_a, L_b)**, and the exchange completes when it ends. The snapshot is taken at **the time the session starts (acceptance)**.
4. When a session starts, the actions in progress on both sides are handled as follows.
   - **The Scorer is cancelled.** The time already spent is lost and there is no result.
   - **Measure continues.** The result goes only into my own notebook and is not delivered to the partner.
   - **A walk is not interrupted.** The session starts after the walk ends (delay of at most t_s).
5. During communication, **moving and the Scorer are impossible.** **Measure is possible** (in place).
6. **Only the two communicating players stop.** Everyone else keeps moving.
7. **Disconnect**: anyone can disconnect midway. On disconnect, nothing is delivered and both sides are released immediately. Time already spent does not come back.
8. There is **no movement merged through communication.** After the communication ends, each player decides how far to go and walks there.
9. If the partner **drops out** mid-session, it is handled the same as a disconnect (nothing delivered, released immediately).

## 6. Information Structure

| Information | Who knows it |
|---|---|
| Every player's L and m (and therefore t_s and s) | Everyone |
| Remaining time, dropout probability distribution, S_max | Everyone |
| Measured elevations | Everyone (dashboard, **updated every 5 seconds**) |
| **Archive** (dropped player's time, position, last measured elevation, L, m) | Everyone |
| My position and descent directions (including the past) | Me |
| Partner snapshots received through communication | Me (notebook) |
| Scorer results | Me (notebook) |
| The true descent direction | Nobody |
| A partner's current position and direction | The partner alone (a snapshot covers only up to that moment) |
| Who is lowest right now (the true score) | Nobody (the dashboard is noisy, and the Scorer only reveals your own) |

### Dashboard

- Every 5 seconds, each player's **most recent measured elevation** (the average over B maps) is published.
- If you measure several times within 5 seconds, only the last value is shown; if you did not measure, the previous value remains.
- When a measurement takes longer than 5 seconds (large B), the value may not change at an update.
- A dropped player's row is frozen at its last value and marked "dropped out."

### Notebook

- Contents: my (position, direction, time), received partner snapshots (position, direction, time), and Scorer records (position, score, time).
- Capacity: **N slots**. When full, you either discard one or **summarize** two records into one.
- Summarize: directions are averaged, positions take the midpoint, and the time is the latest. Precision is lost, but a slot is saved.
- The archive does not use notebook slots (it is a separate public list).

## 7. Lying Variant (optional, off by default)

- **Off**: the snapshot is a true value delivered by the system.
- **On**: the (position, direction) to send during communication is entered by hand and may be false. The recipient records the received value as is and cannot tell whether it is true.
- Means of verification: the dashboard elevation (published by the system, so it cannot be faked), the Scorer, and walking there to check.
- Additional option: a secret **saboteur role** can be mixed in.
- Because the team score is a minimum, when this is on, deceiving the lowest player is the most damaging form of sabotage.

## 8. Noise Near the Bottom

- This is not a separate rule but is produced by **map design**. Each map's bottom is at a slightly different position.
- As you approach the average bottom, the true slope gets smaller while the differences between maps stay the same, so each map points toward its own bottom and the **directions scatter.**
- Result: the closer to the bottom, the larger the noise ratio of the direction measurement.

## 9. Flow of a Game and Strategy Points

**Early game**: the slope is clear. Descend quickly with B=1 and large strides.
**Late game**: the direction wavers. Shrink the stride, raise B, gather more samples through communication, and check often with the Scorer. If there is hardly any improvement, decide whether to go further or stop.
**Endgame**: as T approaches, locking in matters more than new exploration (see "Endgame" below).

- **Time allocation**: a walk costs 1/10 of a measurement, so it is nearly free. The expensive things are Measure, the Scorer (3m), and communication, and the basic question is how to divide time among these three.
- **Averaging samples**: a direction is always a one-map-sized noisy sample, so you have to gather samples and average them to be accurate.
- **Re-measuring in place**: re-measuring at the same spot with B=1 gives the same value, so it wastes time. New samples come from walking (map replacement) or from a larger B.
- **Direction magnitude**: the length of the gradient vector tells you how steep it is. Comparing the gradients at two points lets you estimate curvature and choose a stride (beware of noise).
- **Scorer**: it verifies whether the direction estimate was right. Using it twice, before and after a trial walk, lets you confirm the **true slope** along that direction (cost 2s + t_s = about 6.1m).
- **Checkpoints**: if you record the (position, score) confirmed by the Scorer, you can return to the best point you have seen.
- **Choosing a communication partner**: the cost max(L), the partner's m (sample production rate), whether the partner's position is near mine (a partner's samples are more useful the closer they are), and the dashboard elevation history (coarse, since it updates every 5 seconds).
- **Targeting idle hubs**: low-latency players are popular and often issue 403s. The recipient of a request can choose whether to accept, so requests from slow partners are easily rejected. Communicating with a slow partner ties that player up for a long time and sends 403s to everyone else, so accepting creates an external cost.
- **When a request arrives during the Scorer**: accepting now cancels the Scorer and wastes 3m. Waiting for it to finish may run into the 5-second expiry. A requester also does better to avoid the Scorer while waiting for a response.
- **Distance to walk**: walks are cheap, so the cost of going to a partner is small, but because of S_max a far partner takes several walks. A snapshot goes stale as the partner moves.
- **Measuring during communication**: the waiting time becomes sample-gathering time, which reduces the penalty on players with large L.
- **Disconnect**: a way to cut your losses. You can quit midway even after accepting, but the time already spent is lost to both sides.

### What the Minimum Team Score Creates

- **Leaders and laggards**: only one player needs to reach the lowest point. If laggards are in bad places, the team score is not reduced. Instead, a laggard's job becomes feeding samples to the leader or exploring a different valley.
- **Asymmetric merging**: a communication in which the laggard walks toward the leader and the leader barely moves is advantageous. The leader has no reason to be shaken out of position by a laggard.
- **Diversity becomes valuable**: under an average score, everyone converging on the same place is safe, but under a minimum, trying different places produces the one lucky player.
- **Spotting the leader is hard**: the true score can only be learned by the player themself with the Scorer, and the dashboard is a 5-second update of a B-map average, so it is noisy. Estimating who is leading is part of the game.

### What Dropout Creates

- **Leader insurance**: you don't know when the leader will leave. Since the archive position becomes a score candidate as is, it is important for the leader to always be in a position where it is "fine to leave now."
- **Successor**: keeping a second player near the leader means they can carry on descending if the leader leaves.
- **Picking up from the archive**: walking to a dropped player's last position and checking it with the Scorer is effectively a free checkpoint. There is no direction information, so from there you have to measure for yourself.
- **Dropout during communication**: the partner you were just talking with can vanish. A long communication is also exposed to dropout risk.

### Endgame

- The score is the **true score of the position at time T**. It is safest to return to the best Scorer-verified point and meet T there.
- Any walk, measurement, or communication that does not finish by T is wasted. Start the final actions only after calculating that they will finish before T.
- A lucky dashboard value is not a score, so in the endgame Scorer verification becomes more important.

## 10. Extension Options

- Number of dimensions: 2D → several hundred dimensions.
- Balance: give a correction in which a higher L comes with more accurate measurement.
- Revert the Scorer time s to a constant common to all players.
- Publish Scorer results on the dashboard (makes spotting the leader and follower strategies easier).
- Include the score in snapshots.
- Allow the Scorer during communication (a larger overlap).
- Change the team-score aggregation: minimum / mean / mean of the top k players. (The mean is closer to real distributed training, which "builds one consensus model." For the purpose of extracting hints, run both aggregations.)
- Raise the dropout probability, or leave part of the notebook in the archive at the moment of dropout.
- Give the descent direction as a unit vector (remove magnitude).
- Start scattered (different initial values).
- Make every measurement draw B fresh maps (remove the fixed current map).

## 11. Correspondence Table to Real Distributed Training

In the "Translation" column, **direct** means the result can be carried straight over to a real rule, **indirect** means idea-level only, and **fun-only** means there is no correspondence and it must not be used as a hint.

| Game | Distributed training | Translation |
|---|---|---|
| Position on the mountain | Model parameters (a vector) | Direct |
| 100 maps | Data (each a slightly different loss) | Direct |
| Current map, replaced at every walk | Current minibatch, a new batch every step | Direct |
| Re-measuring in place gives the same value | Recomputing the gradient on the same batch gives the same result | Direct |
| Measure (elevation + gradient vector) | Minibatch loss + gradient | Direct |
| Batch size B | Batch size | Direct |
| Scorer (cost 3m) | Full-dataset loss / validation loss (proportional to compute speed) | Direct |
| True descent direction is unknown | The full gradient is not computed | Direct |
| Walk, stride | Optimizer step, learning rate | Direct |
| t_s = m/10 | An optimizer step is far cheaper than computing a gradient | Direct |
| S_max | Maximum update size per step (clipping) | Indirect |
| m | Gradient computation time (GPU performance differences) | Direct |
| L, stopping during communication | Network latency, blocking communication | Direct |
| Measuring allowed during communication | Communication–computation overlap | Direct |
| One communication at a time, 403 if busy | Handling only one communication at a time (busy rejection) | Direct |
| Mutual acceptance, 5-second request expiry | Connection request → accept handshake, timeout | Direct |
| Scorer cancelled at session start | Interrupting an in-progress evaluation job when communication starts | Indirect |
| Disconnect | Timeout, connection loss | Direct |
| Notebook, summarize | Past-gradient buffer (momentum), memory constraints | Indirect |
| Each player decides how far to move after communication | How much of the received parameters to mix in (mixing ratio) | Direct |
| Everyone starts from the same point | Same initialization | Direct |
| Lying variant | Malicious (Byzantine) nodes | Direct |
| Directions scatter near the bottom | Gradient noise ratio rises near convergence | Direct |
| Late game: stride↓ B↑ | Learning rate decay, batch size increase | Direct |
| Dashboard 5-second update | Periodic reporting of metrics | Indirect |
| Time limit T | Training budget (wall-clock time) | Direct |
| Team score = lowest player | Adopting the best checkpoint among several replicas as the final model (best-of-N) | Indirect (differs from the consensus-model goal) |
| Random dropout | Node departure, preemption, crash (churn) | Direct |
| Archive | The last checkpoint and metadata a departed node leaves behind | Indirect |
| Walking to an archive position | Continuing training from another node's checkpoint | Direct |
| Leaders and laggards | Concentrating training on one replica while the others supply samples and exploration | Indirect |
| Scorer results private, no score in snapshots | (In reality, the validation loss is usually sent along) | Fun-only |
| 100 maps, discrete map replacement | Finite dataset | Fun-only |

## 12. Things Still Held as Assumptions

### End, Score, Dropout

- The team score includes dropped-out players, evaluated at the **true score** of the archive position.
- The archive is public, and its entries are the dropout time, position, last measured elevation, L, and m. If the player took a walk just before dropping out, the last measured elevation may be a **value measured at the previous position**, and it is noisy (it is not an exact score).
- Dropout is independent per player and there is no re-entry. The shape of the dropout distribution and its mean (mean 2T) are placeholders.
- At the moment of dropout, the action in progress is discarded and the position is that of the last completed walk.
- Actions that have not finished at time T are void.

### Numbers

- All numeric values (T, S_max, B_max, N, m, the L range, dashboard period, request expiry, number of players) are placeholder examples.
- S_max defaults to "1/3 of the distance from the start to the average bottom" and is used as a dial to adjust balance.

### Communication

- Scorer results are private, and communication snapshots do not include scores.
- The Scorer is impossible during communication.
- A 403 returns immediately and costs no time. Whether someone is in communication is not public (you only find out by requesting).
- Accept / reject responses are delivered immediately (L applies only to session time). This holds even if the other player is doing something else.
- If several requests arrive at one player at once, only the first one accepted takes effect, and the rest get 403.
- Movement caused by communication does not change the map.
- A disconnect is detected by the other side immediately (no notification delay). Summarize takes no time.

### Measurement

- The B−1 maps other than the current map are drawn fresh at each measurement (the current map is fixed until a walk).
- The descent direction includes magnitude.

### Information

- The dashboard publishes the **most recent measured value** every 5 seconds (not an average over the interval), and the measurement time is not shown.

### Other

- The lying variant is off by default.
- When humans play directly, operation speed can mask L. L needs to be raised to 1–10 seconds, or the game speed lowered. (Not applicable to RL experiments.)

---

## Changelog

**v0.5**
- Stated the purpose explicitly as "hints for distributed-training protocols," and added a "Translatability" column to the correspondence table (direct / indirect / fun-only).
- Fixed which maps a measurement draws: the current map + B−1 random maps. Re-measuring in place with B=1 gives the same value.
- Introduced the maximum distance of one walk, S_max (1/3 of the distance from the start to the average bottom).
- Included magnitude in the descent direction (a gradient vector).
- Requests and responses are always delivered immediately. At session start the Scorer is cancelled, Measure continues, and a walk finishes before the session starts (communication rules 2 and 4).
- Fixed the starting position as the same point for everyone.
- Changed the Scorer time from a common constant to **s = 3m** (per player).
- Fixed 6 players and the 5-second request expiry as placeholder values. The dropout distribution is unchanged from v0.4.
- Updated extension options (running aggregation modes in parallel, unit-vector direction, scattered start, fresh-draw measurement every time).
- Resolved all "TBD" items and organized them into "Things Still Held as Assumptions."

**v0.4**
- Fixed the end condition as time limit T (value is a placeholder).
- Changed the team score from "mean over everyone" to "the **true score of the lowest player at time T**."
- Added **random dropout** and the **public archive**. Dropped players are included as score candidates at their archive position.
- Dropout during communication is handled the same as a disconnect.
- Added remaining time, the archive, and the dropout distribution to the information structure. Added a dropout marker to the dashboard.
- Added a strategy section: leaders/laggards under a minimum score, dropout insurance, and endgame play.
- Added T, best-of-N, dropout, and the archive to the correspondence table.

**v0.3**
- Restored mutual acceptance (request → partner accepts → session starts. Players move freely before acceptance).

**v0.2**
- Changed walk time to m/10 (a per-player value).
- Removed simultaneous communication. A player in communication returns 403.
- Removed anti-abuse rules: mutual acceptance (restored in v0.3), the proposed re-request cooldown, and the proposal to make the in-communication state public.
- Removed values related to simultaneous communication (τ, the minimum limit on L).
- Changed dashboard elevation publication from "immediately on measurement" to "updated every 5 seconds."
- Stated L explicitly as a fixed value.
