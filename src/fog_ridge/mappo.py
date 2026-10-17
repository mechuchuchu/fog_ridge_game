"""Shared decentralized actor / centralized set critic for asynchronous solo-action teams."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .baseline import DecayingSGD
from .policy import KINDS, ActorCritic, SetBlock, categorical, decode_action, pack, privileged
from .ppo import gae, save_checkpoint, stack, update
from .simulator import Action, Config
from .team import COMM_KINDS, TeamEnv

TEAM_KINDS = KINDS + COMM_KINDS


def pack_team(env, player, capacity, device="cpu"):
    data = pack(env.agents[player].observe(), capacity, device)
    known = [a.dashboard for a in env.agents if a.dashboard is not None]
    reference = max(min(known), 1e-12) if known else 1
    incoming = {r.sender: r for r in env.incoming(player)}
    outgoing = env.outgoing(player)
    session = env.session(player)
    public = [
        [
            np.log1p(a.config.measurement_time),
            np.log1p(a.config.latency),
            float(a.dashboard is not None),
            np.log1p((a.dashboard or 0) / reference),
            np.log1p(env.time - a.dashboard_time),
            float(i == player),
            float(i in incoming),
            float(outgoing is not None and outgoing.target == i),
            float(session is not None and i != player and i in (session.a, session.b)),
            (incoming[i].expires - env.time) / env.request_timeout if i in incoming else 0.0,
        ]
        for i, a in enumerate(env.agents)
    ]
    data["players"] = torch.tensor(public, dtype=torch.float32, device=device).unsqueeze(0)
    mask = env.action_mask(player)
    data["legal"] = torch.tensor([[mask[k] for k in TEAM_KINDS]], device=device)
    if not env.agents[player].records:
        data["legal"][0, KINDS.index("walk")] = False
    data["targets"] = torch.tensor(
        np.array([env.target_mask(player, k) for k in ("request", "accept", "reject", "ignore")]),
        device=device,
    ).unsqueeze(0)
    remaining = env.futures[player][0].time - env.time if player in env.futures else 0
    local_kind = env.futures[player][1]["action"] if player in env.futures else ""
    local = [
        np.log1p(remaining),
        float(local_kind == "measure"),
        float(local_kind == "score"),
        float(local_kind == "walk"),
        float(session is not None),
        float(session is not None and session.start is None),
        np.log1p(env.time - session.start) if session and session.start is not None else 0,
        np.log1p(max(0, session.end - env.time)) if session and session.start is not None else 0,
        float(env.last_result[player] == "403"),
    ]
    data["communication"] = torch.tensor([local], dtype=torch.float32, device=device)
    sources = np.zeros((capacity, 3), dtype=np.float32)
    for i, record in enumerate(env.agents[player].records):
        sources[i] = [
            float(record.kind == "snapshot"),
            np.log1p(record.source_m),
            np.log1p(record.source_l),
        ]
    data["sources"] = torch.tensor(sources, device=device).unsqueeze(0)
    return data


def central_state(env, player, device="cpu"):
    focal = env.agents[player]
    tokens = []
    for i, agent in enumerate(env.agents):
        remaining = env.futures[i][0].time - env.time if i in env.futures else 0
        tokens.append(
            np.r_[
                privileged(agent),
                np.log1p(agent.config.measurement_time),
                np.log1p(agent.config.latency),
                np.log1p(remaining),
                np.log1p(np.linalg.norm(agent.position - focal.position)),
                float(i in env.links),
                np.log1p(max(0, env.session(i).end - env.time))
                if env.session(i) is not None and env.session(i).start is not None
                else 0,
            ]
        )
    # Focal query first; following player tokens form an unordered set.
    return torch.tensor(
        np.array([tokens[player], *tokens]), dtype=torch.float32, device=device
    ).unsqueeze(0)


class CentralCritic(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(14, width), nn.Tanh())
        self.blocks = nn.ModuleList([SetBlock(width), SetBlock(width)])
        self.head = nn.Sequential(nn.Linear(2 * width, width), nn.Tanh(), nn.Linear(width, 1))

    def forward(self, state):
        focal = self.embed(state[:, 0])
        x = self.embed(state[:, 1:])
        valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        pair = torch.zeros(*x.shape[:2], x.shape[1], 2, device=x.device)
        for block in self.blocks:
            x = block(x, valid, pair)
        return self.head(torch.cat([focal, x.mean(1)], -1))


class MAPPO(ActorCritic):
    def __init__(self, capacity=20, batch_max=8, width=64):
        super().__init__(capacity, batch_max, width)
        self.public = nn.Sequential(nn.Linear(10, width), nn.Tanh(), nn.Linear(width, width))
        self.communication = nn.Sequential(nn.Linear(9, width), nn.Tanh())
        self.source_embed = nn.Linear(3, width, bias=False)
        self.kind = nn.Linear(width, len(TEAM_KINDS))
        # Local behavior dominates before the communication curriculum/BC warm start.
        with torch.no_grad():
            self.kind.bias[len(KINDS) :].fill_(-2)
        self.partner_query = nn.Linear(width, width)
        self.critic = CentralCritic(width)

    def actor_context(self, data):
        root = super().actor_context(data)
        if "players" in data:
            root = root + self.public(data["players"]).mean(1)
            root = root + self.communication(data["communication"])
        return root

    def token_embeddings(self, data):
        tokens = super().token_embeddings(data)
        return tokens + self.source_embed(data["sources"]) if "sources" in data else tokens

    def act(self, data, actions=None, deterministic=False):
        if "players" not in data:
            # Solo SGD warm start keeps communication unavailable.
            data = dict(data)
            data["legal"] = torch.cat(
                [
                    data["legal"],
                    torch.zeros(
                        len(data["legal"]),
                        len(COMM_KINDS),
                        dtype=torch.bool,
                        device=data["legal"].device,
                    ),
                ],
                -1,
            )
            return super().act(data, actions, deterministic)
        result, logp, entropy = super().act(
            data, None if actions is None else actions[:, :8], deterministic
        )
        root, _ = self.encode(data)
        logits = torch.einsum(
            "bw,bpw->bp", self.partner_query(root), self.public(data["players"])
        ) / np.sqrt(root.shape[-1])
        kind = result[:, 0].long()
        branch = (kind - len(KINDS)).clamp(0, 3)
        masks = data["targets"][torch.arange(len(kind), device=kind.device), branch]
        distribution = categorical(logits, masks)
        target = (
            actions[:, 8].long()
            if actions is not None
            else (distribution.logits.argmax(-1) if deterministic else distribution.sample())
        )
        active = (kind >= len(KINDS)) & (kind < len(KINDS) + 4)
        logp = logp + distribution.log_prob(target) * active
        entropy = entropy + 0.1 * distribution.entropy() * active
        target = torch.where(active, target, torch.zeros_like(target))
        return torch.cat([result, target[:, None].float()], -1), logp, entropy


def decode_team(env, player, sample):
    kind = int(sample[0].item())
    if kind < len(KINDS):
        return decode_action(env.agents[player], sample[:8])
    return Action(TEAM_KINDS[kind], target=int(sample[8].item()))


class GossipTeacher:
    """SGD with periodic handshakes; no peer private state is used for decisions."""

    def __init__(self, env, interval=5.0):
        self.policies = [DecayingSGD() for _ in env.agents]
        self.next_request = [2 * a.config.measurement_time for a in env.agents]
        self.interval = interval
        self.rounds = [0 for _ in env.agents]

    def act(self, env, player):
        local, mask = env.agents[player], env.action_mask(player)
        if local.pending is not None:
            return self.policies[player].act(local.observe(), local.compose_walk)
        if mask["accept"]:
            request = min(env.incoming(player), key=lambda r: r.expires)
            return Action("accept", target=request.sender)
        if player in env.futures:
            return Action("continue")
        if mask["request"] and local.records and env.time >= self.next_request[player]:
            peers = [i for i in range(len(env.agents)) if i != player]
            # Cycle through public peers to avoid teaching a single hub convention.
            target = peers[self.rounds[player] % len(peers)]
            self.rounds[player] += 1
            self.next_request[player] = env.time + self.interval
            return Action("request", target=target)
        policy = self.policies[player]
        if player in env.links:
            return Action("measure")
        return policy.act(local.observe(), local.compose_walk)


def imitate_team(
    model, optimizer, config, seeds, players, aggregation, device, epochs=20, communication=True
):
    observations, targets = [], []
    totals = None
    for seed in seeds:
        env = TeamEnv(config, players, seed, aggregation, communication=communication)
        teacher = GossipTeacher(env)
        for _ in range(100_000 * players):
            player = env.next_agent()
            if player is None:
                break
            data = pack_team(env, player, model.settings["capacity"], device)
            policy = teacher.policies[player]
            steps = policy.steps
            action = teacher.act(env, player)
            target = torch.zeros(1, 9, device=device)
            target[0, 0] = TEAM_KINDS.index(action.kind)
            target[0, 1] = action.batch - 1
            target[0, 6] = np.log(policy.learning_rate / (1 + policy.decay * steps))
            target[0, 7] = -8
            if action.kind == "walk":
                target[0, 2] = np.flatnonzero(data["tokens"][0, :, 0].cpu().numpy() == 1)[-1]
            if action.slots:
                target[0, 4] = action.slots[0]
            if action.kind in ("request", "accept", "reject", "ignore"):
                target[0, 8] = action.target
            observations.append(data)
            targets.append(target)
            env.submit(player, action)
        else:
            raise RuntimeError("communication teacher decision budget exceeded")
        if totals is None:
            totals = dict.fromkeys(env.stats, 0)
        for key in env.stats:
            totals[key] += env.stats[key]
    if not observations:
        return {}
    data, targets = stack(observations), torch.cat(targets)
    for _ in range(epochs):
        for ids in torch.randperm(len(targets), device=device).split(128):
            _, logp, _ = model.act({k: v[ids] for k, v in data.items()}, targets[ids])
            loss = -logp.mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5, error_if_nonfinite=True)
            optimizer.step()
    return totals


def collect_team(
    model,
    config,
    seeds,
    players=4,
    aggregation="min",
    device="cpu",
    gamma=1.0,
    lam=0.95,
    deterministic=False,
    communication=True,
):
    """Close a transition at this agent's next decision, not at a teammate's event."""
    data_rows, critic_rows, action_rows, logps, advantages, returns, reports = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for seed in seeds:
        env = TeamEnv(config, players, seed, aggregation, communication=communication)
        trajectories = [[] for _ in env.agents]
        pending = {}
        initial = env.potential

        def close(player, pending=pending, env=env, trajectories=trajectories):
            row = pending.pop(player)
            row["reward"] = env.potential - row.pop("potential")
            row["dt"] = env.time - row.pop("start")
            trajectories[player].append(row)

        for _ in range(100_000 * players):
            player = env.next_agent()
            if player is None:
                for i in list(pending):
                    close(i)
                break
            if player in pending:
                close(player)
            data = pack_team(env, player, model.settings["capacity"], device)
            critic = central_state(env, player, device)
            with torch.no_grad():
                action, logp, _ = model.act(data, deterministic=deterministic)
                value = model.value(critic).item()
            pending[player] = {
                "data": data,
                "critic": critic,
                "action": action,
                "logp": logp,
                "value": value,
                "potential": env.potential,
                "start": env.time,
            }
            env.submit(player, decode_team(env, player, action[0]))
        else:
            raise RuntimeError("team decision budget exceeded")
        rewards_by_agent = []
        for trajectory in trajectories:
            adv, ret = gae(
                [r["reward"] for r in trajectory],
                [r["value"] for r in trajectory],
                [r["dt"] for r in trajectory],
                gamma,
                lam,
            )
            advantages.extend(adv)
            returns.extend(ret)
            for row in trajectory:
                data_rows.append(row["data"])
                critic_rows.append(row["critic"])
                action_rows.append(row["action"])
                logps.append(row["logp"])
            rewards_by_agent.append(float(sum(r["reward"] for r in trajectory)))
        reports.append(
            {
                "seed": seed,
                "gap": env.gap,
                "player_gaps": env.gaps.tolist(),
                "potential_change": env.potential - initial,
                "agent_rewards": rewards_by_agent,
                "decisions": [len(t) for t in trajectories],
                "communication": dict(env.stats),
                "communication_events": [row for row in env.trace if "event" in row],
                "players": [
                    {"L": a.config.latency, "m": a.config.measurement_time} for a in env.agents
                ],
            }
        )
    return {
        "data": stack(data_rows),
        "critic": torch.cat(critic_rows),
        "actions": torch.cat(action_rows),
        "logp": torch.cat(logps),
        "advantages": torch.tensor(advantages, device=device),
        "returns": torch.tensor(returns, device=device),
    }, reports


def baseline_team(config, seed, players, aggregation):
    env = TeamEnv(config, players, seed, aggregation)
    policies = [DecayingSGD() for _ in env.agents]
    for _ in range(100_000 * players):
        player = env.next_agent()
        if player is None:
            return env.gap
        local = env.agents[player]
        env.submit(player, policies[player].act(local.observe(), local.compose_walk))
    raise RuntimeError("baseline decision budget exceeded")


def evaluate_team(
    model,
    config,
    seeds,
    players,
    aggregation,
    device="cpu",
    communication=True,
    deterministic=False,
):
    results = []
    cuda_devices = [torch.cuda.current_device()] if torch.device(device).type == "cuda" else []
    for seed in seeds:
        # Evaluation is reproducible per world and does not consume the training RNG.
        with torch.random.fork_rng(devices=cuda_devices):
            torch.random.default_generator.manual_seed(seed)
            if cuda_devices:
                torch.cuda.manual_seed(seed)
            _, reports = collect_team(
                model,
                config,
                [seed],
                players,
                aggregation,
                device,
                deterministic=deterministic,
                communication=communication,
            )
        results.append(
            {
                "seed": seed,
                "mappo_gap": reports[0]["gap"],
                "sgd_gap": baseline_team(config, seed, players, aggregation),
                "communication": reports[0]["communication"],
                "communication_events": reports[0]["communication_events"],
            }
        )
    return {
        "deterministic": deterministic,
        "episodes": results,
        **{
            f"{name}_median_log10_gap": float(
                np.median([np.log10(row[f"{name}_gap"] + 1e-12) for row in results])
            )
            for name in ("mappo", "sgd")
        },
    }


def load_mappo(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if (
        checkpoint.get("algorithm") != "mappo"
        or checkpoint["format_version"] != 1
        or checkpoint.get("communication_version") != 1
    ):
        raise ValueError("expected a MAPPO checkpoint")
    model = MAPPO(**checkpoint["model_settings"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, Config(**checkpoint["config"]), checkpoint


def save_mappo(path, model, optimizer, config, iteration, next_seed, training):
    save_checkpoint(
        path,
        model,
        optimizer,
        config,
        iteration,
        next_seed,
        training,
        algorithm="mappo",
        extra={"communication_version": 1},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-communication", action="store_true", default=None, help="communication ablation"
    )
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--aggregation", choices=("min", "mean"), default="min")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--episodes-per-update", type=int, default=4)
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--bc-episodes", type=int, default=4)
    parser.add_argument("--bc-epochs", type=int, default=20)
    parser.add_argument(
        "--deterministic-evaluation",
        action="store_true",
        help="argmax evaluation instead of reproducible policy sampling",
    )
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("results/mappo"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--evaluate", type=Path)
    args = parser.parse_args()
    if (
        min(
            args.players,
            args.iterations,
            args.episodes_per_update,
            args.epochs,
            args.threads,
            args.eval_episodes,
            args.bc_epochs,
        )
        < 1
        or args.bc_episodes < 0
        or not 0 < args.gamma <= 1
        or not 0 <= args.gae_lambda <= 1
        or not np.isfinite(args.learning_rate)
        or args.learning_rate <= 0
    ):
        parser.error("invalid MAPPO settings")
    if args.resume and args.evaluate:
        parser.error("choose either resume or evaluate")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = args.device
    config = Config(dimension=args.dimension, duration=args.duration)
    iteration, next_seed = 0, args.seed
    if args.resume or args.evaluate:
        model, config, checkpoint = load_mappo(args.resume or args.evaluate, device)
        iteration, next_seed = checkpoint["iteration"], checkpoint["next_seed"]
        for key, value in checkpoint["training"].items():
            if key != "no_communication" or args.no_communication is None:
                setattr(args, key, value)
    else:
        model = MAPPO(config.notebook_size, config.batch_max).to(device)
    args.no_communication = bool(args.no_communication)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if device == "cuda" and checkpoint["cuda_rng"]:
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint["cuda_rng"]])
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.resume and not args.evaluate:
        bc_stats = imitate_team(
            model,
            optimizer,
            config,
            range(next_seed, next_seed + args.bc_episodes),
            args.players,
            args.aggregation,
            device,
            args.bc_epochs,
            communication=not args.no_communication,
        )
        (args.output / "imitation.json").write_text(json.dumps(bc_stats, indent=2) + "\n")
        next_seed += args.bc_episodes
    training = {
        k: getattr(args, k)
        for k in (
            "players",
            "aggregation",
            "gamma",
            "gae_lambda",
            "epochs",
            "episodes_per_update",
            "no_communication",
        )
    }
    if not args.evaluate:
        for step in range(iteration + 1, iteration + args.iterations + 1):
            batch, reports = collect_team(
                model,
                config,
                range(next_seed, next_seed + args.episodes_per_update),
                args.players,
                args.aggregation,
                device,
                args.gamma,
                args.gae_lambda,
                communication=not args.no_communication,
            )
            next_seed += args.episodes_per_update
            metrics = update(model, optimizer, batch, epochs=args.epochs)
            entry = {
                "iteration": step,
                "transitions": len(batch["actions"]),
                "communication": {
                    key: sum(r["communication"][key] for r in reports)
                    for key in reports[0]["communication"]
                },
                "median_log10_gap": float(np.median([np.log10(r["gap"] + 1e-12) for r in reports])),
                **metrics,
            }
            with (args.output / "train.jsonl").open("a") as stream:
                stream.write(json.dumps(entry) + "\n")
            with (args.output / "episodes.jsonl").open("a") as stream:
                for report in reports:
                    stream.write(json.dumps({"iteration": step, **report}) + "\n")
            print(json.dumps(entry), flush=True)
            save_mappo(
                args.output / "checkpoint.pt", model, optimizer, config, step, next_seed, training
            )
    result = evaluate_team(
        model,
        config,
        range(1_000_000 + args.seed, 1_000_000 + args.seed + args.eval_episodes),
        args.players,
        args.aggregation,
        device,
        communication=not args.no_communication,
        deterministic=args.deterministic_evaluation,
    )
    (args.output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "run.json").write_text(
        json.dumps(
            {
                "environment": asdict(config),
                "training": training,
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            },
            indent=2,
        )
        + "\n"
    )
    summary = {k: v for k, v in result.items() if k != "episodes"}
    summary["evaluation_episodes"] = len(result["episodes"])
    summary["communication"] = {
        key: sum(row["communication"][key] for row in result["episodes"])
        for key in result["episodes"][0]["communication"]
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
