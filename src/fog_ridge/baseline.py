"""Untuned decaying SGD using only actor observations and the action decoder."""

import numpy as np

from .simulator import Action


class DecayingSGD:
    def __init__(self, learning_rate=0.08, decay=0.02, batch=1, score_every=20):
        if learning_rate <= 0 or decay < 0 or batch < 1 or score_every < 1:
            raise ValueError("invalid SGD parameters")
        self.learning_rate = learning_rate
        self.decay = decay
        self.batch = batch
        self.score_every = score_every
        self.steps = 0
        self.phase = "measure"

    def act(self, observation, decode):
        mask = observation["mask"]
        records = observation["records"]
        if not mask["measure"]:
            # Forced eviction; preserve the best checkpoint when possible.
            candidates = np.flatnonzero(records[:, 1] == 0)
            slot = int(candidates[0]) if len(candidates) else int(np.argmax(records[:, 8]))
            return Action("delete", slots=(slot,))
        if self.phase == "walk":
            self.phase = "score" if (self.steps + 1) % self.score_every == 0 else "measure"
            weights = np.zeros(len(records))
            candidates = np.flatnonzero(records[:, 0] == 1)
            if not len(candidates):
                self.phase = "measure"
                return Action("measure", batch=self.batch)
            weights[candidates[-1]] = 1
            rate = self.learning_rate / (1 + self.decay * self.steps)
            self.steps += 1
            return decode(weights, np.zeros(len(records)), gradient_scale=rate)
        if self.phase == "score":
            self.phase = "measure"
            return Action("score")
        self.phase = "walk"
        return Action("measure", batch=self.batch)


def rollout(env, policy, max_decisions=1_000_000):
    initial_gap = env.landscape.gap(env.position)
    total_reward = 0.0
    for _ in range(max_decisions):
        if env.done:
            break
        action = policy.act(env.observe(), env.compose_walk)
        _, reward, _, _ = env.step(action)
        total_reward += reward
    else:
        raise RuntimeError("decision budget exceeded (possible zero-time action loop)")
    return {
        "initial_gap": initial_gap,
        "final_gap": env.landscape.gap(env.position),
        "score": env.landscape.score(env.position),
        "reward": total_reward,
        "time": env.time,
        "events": len(env.trace),
    }
