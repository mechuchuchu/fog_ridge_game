"""Coordinate-free set actor and separate privileged scalar critic for solo PPO."""

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal

from .simulator import Action

KINDS = ("measure", "score", "walk", "wait", "delete", "summarize")


def pack(observation, capacity, device="cpu"):
    """Normalize actor-visible features only; pad notebook without adding slot IDs."""
    records = observation["records"].copy()
    n = len(records)
    if n > capacity:
        raise ValueError("notebook exceeds policy capacity")
    if n:
        records[:, 3] /= max(observation["time"] + 1, 1)
        records[:, 4:7] = np.log1p(records[:, 4:7])
        scores = records[:, 1].astype(bool)
        reference = max(float(np.min(records[scores, 8])), 1e-12) if scores.any() else 1
        records[:, 8] = np.log1p(records[:, 8] / reference)
    tokens = np.zeros((capacity, 9), dtype=np.float32)
    tokens[:n] = records
    valid = np.arange(capacity) < n
    pair = np.zeros((capacity + 1, capacity + 1, 2), dtype=np.float32)
    pair[1 : n + 1, 1 : n + 1, 0] = observation["direction_cosines"]
    pair[1 : n + 1, 1 : n + 1, 1] = observation["direction_offset_cosines"]
    dashboard = observation["dashboard"]
    context = np.array(
        [
            observation["remaining"],
            np.log1p(observation["time"]),
            np.log1p(observation["m"]),
            np.log1p(observation["L"]),
            np.log1p(observation["step_max"]),
            float(dashboard is not None),
            np.log1p(max(dashboard or 0, 0)),
            np.log1p(observation["dashboard_age"]),
            observation["free_slots"] / capacity,
        ],
        dtype=np.float32,
    )
    data = {
        "tokens": tokens,
        "valid": valid,
        "pair": pair,
        "context": context,
        "legal": np.array([observation["mask"][k] for k in KINDS]),
    }
    # This decoder has no exploration-vector branch: an empty notebook cannot move.
    if n == 0:
        data["legal"][KINDS.index("walk")] = False
    return {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in data.items()}


def privileged(env):
    world, x = env.landscape, env.position
    return np.array(
        [
            env.time / env.config.duration,
            np.log1p(world.gap(x)),
            np.log1p(np.linalg.norm(x - world.center)),
            np.log1p(np.linalg.norm(world.curvature * (x - world.center))),
            np.log1p(world.curvature.min()),
            np.log1p(world.curvature.max()),
            np.log1p(world.curvature.mean()),
            np.log1p(world.floor),
        ],
        dtype=np.float32,
    )


class SetBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.qkv = nn.Linear(width, 3 * width)
        self.bias = nn.Linear(2, 1, bias=False)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.Tanh(), nn.Linear(2 * width, width))

    def forward(self, x, valid, pair):
        q, k, v = self.qkv(x).chunk(3, -1)
        logits = q @ k.transpose(-1, -2) / np.sqrt(x.shape[-1])
        logits = logits + self.bias(pair).squeeze(-1)
        weights = logits.masked_fill(~valid[:, None, :], -1e9).softmax(-1)
        x = self.norm1(x + weights @ v)
        return self.norm2(x + self.ff(x))


def categorical(logits, mask):
    # Inactive conditional heads need a finite dummy distribution too.
    mask = mask.clone()
    mask[~mask.any(-1), 0] = True
    return Categorical(logits=logits.masked_fill(~mask, -1e9))


class ActorCritic(nn.Module):
    def __init__(self, capacity=20, batch_max=8, width=64):
        super().__init__()
        self.settings = {"capacity": capacity, "batch_max": batch_max, "width": width}
        self.embed = nn.Linear(9, width)
        self.context = nn.Sequential(nn.Linear(9, width), nn.Tanh())
        self.blocks = nn.ModuleList([SetBlock(width), SetBlock(width)])
        self.kind = nn.Linear(width, len(KINDS))
        self.batch = nn.Linear(width, batch_max)
        self.pointers = nn.Linear(width, 4 * width)
        self.scales = nn.Linear(width, 2)
        nn.init.zeros_(self.scales.weight)
        self.scales.bias.data.copy_(torch.tensor([-2.5, -5.0]))
        self.log_std = nn.Parameter(torch.full((2,), -0.7))
        self.critic = nn.Sequential(
            nn.Linear(8, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1)
        )

    def value(self, critic):
        return self.critic(critic).squeeze(-1)

    def act(self, data, actions=None, deterministic=False):
        valid = data["valid"]
        root = self.context(data["context"])
        x = torch.cat([root[:, None], self.embed(data["tokens"])], 1)
        full_valid = torch.cat([torch.ones_like(valid[:, :1]), valid], 1)
        for block in self.blocks:
            x = block(x, full_valid, data["pair"])
        root, tokens = x[:, 0], x[:, 1:]
        queries = self.pointers(root).reshape(-1, 4, root.shape[-1])
        logits = torch.einsum("bhw,bnw->bhn", queries, tokens) / np.sqrt(root.shape[-1])
        direction = valid & ((data["tokens"][:, :, 0] + data["tokens"][:, :, 2]) > 0)
        distributions = [
            categorical(self.kind(root), data["legal"]),
            Categorical(logits=self.batch(root)),
            categorical(logits[:, 0], direction),
            categorical(logits[:, 1], valid),
        ]
        chosen = []
        for i, distribution in enumerate(distributions):
            chosen.append(
                actions[:, i].long()
                if actions is not None
                else distribution.logits.argmax(-1)
                if deterministic
                else distribution.sample()
            )
        kind = chosen[0]
        management_mask = torch.where((kind == 5)[:, None], direction, valid)
        first = categorical(logits[:, 2], management_mask)
        s1 = (
            actions[:, 4].long()
            if actions is not None
            else (first.logits.argmax(-1) if deterministic else first.sample())
        )
        second_mask = direction.clone()
        second_mask.scatter_(1, s1[:, None], False)
        second = categorical(logits[:, 3], second_mask)
        s2 = (
            actions[:, 5].long()
            if actions is not None
            else (second.logits.argmax(-1) if deterministic else second.sample())
        )
        scales = Normal(self.scales(root), self.log_std.clamp(-3, 1).exp())
        z = (
            actions[:, 6:8]
            if actions is not None
            else (scales.mean if deterministic else scales.sample())
        )
        result = torch.cat([torch.stack(chosen + [s1, s2], -1).float(), z], -1)
        logp = distributions[0].log_prob(kind)
        entropy = distributions[0].entropy()
        # Joint log likelihood includes only parameters used by the selected action.
        for distribution, choice, active in [
            (distributions[1], chosen[1], kind == 0),
            (distributions[2], chosen[2], (kind == 2) & direction.any(-1)),
            (distributions[3], chosen[3], (kind == 2) & valid.any(-1)),
            (first, s1, (kind == 4) | (kind == 5)),
            (second, s2, kind == 5),
        ]:
            logp = logp + distribution.log_prob(choice) * active
            entropy = entropy + 0.1 * distribution.entropy() * active
        active_scales = torch.stack(
            [(kind == 2) & direction.any(-1), (kind == 2) & valid.any(-1)], -1
        )
        logp = logp + (scales.log_prob(z) * active_scales).sum(-1)
        entropy = entropy + 0.1 * (scales.entropy() * active_scales).sum(-1)
        return result, logp, entropy


def decode_action(env, sample):
    values = sample.detach().cpu().numpy()
    kind = KINDS[int(values[0])]
    if kind == "walk":
        n = len(env.records)
        gw, pw = np.zeros(n), np.zeros(n)
        g, p = int(values[2]), int(values[3])
        if n:
            if env.records[g].direction is not None:
                gw[g] = 1
            pw[p] = 1
        return env.compose_walk(
            gw,
            pw,
            gradient_scale=float(np.exp(np.clip(values[6], -12, 4))),
            position_scale=float(np.exp(np.clip(values[7], -12, 4))),
        )
    if kind in ("delete", "summarize"):
        slots = (int(values[4]),) if kind == "delete" else (int(values[4]), int(values[5]))
        return Action(kind, slots=slots)
    return Action(kind, batch=int(values[1]) + 1)
