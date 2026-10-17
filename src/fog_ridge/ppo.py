"""Solo PPO with complete-episode rollouts and semi-Markov GAE."""

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .baseline import DecayingSGD, rollout
from .policy import KINDS, ActorCritic, decode_action, pack, privileged
from .simulator import Config, SoloEnv


def gae(rewards, values, durations, gamma=1.0, lam=0.95):
    """Complete episode: terminal bootstrap is zero, discount is per second.

    Lambda is per decision (including instantaneous notebook decisions).
    """
    advantages = np.zeros(len(rewards), dtype=np.float32)
    carry, next_value = 0.0, 0.0
    for i in reversed(range(len(rewards))):
        discount = gamma ** durations[i]
        delta = rewards[i] + discount * next_value - values[i]
        carry = delta + discount * lam * carry
        advantages[i] = carry
        next_value = values[i]
    return advantages, advantages + np.asarray(values, dtype=np.float32)


def stack(items):
    return {key: torch.cat([item[key] for item in items]) for key in items[0]}


def collect(model, config, seeds, device, gamma=1.0, lam=0.95):
    observations, critics, actions, old_logps, all_adv, all_returns, episodes = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for seed in seeds:
        env = SoloEnv(config, seed)
        rewards, values, durations = [], [], []
        initial = env.landscape.gap(env.position)
        for _ in range(100_000):
            data = pack(env.observe(), model.settings["capacity"], device)
            critic = torch.tensor(privileged(env), device=device).unsqueeze(0)
            with torch.no_grad():
                action, logp, _ = model.act(data)
                value = model.value(critic)
            _, reward, done, info = env.step(decode_action(env, action[0]))
            observations.append(data)
            critics.append(critic)
            actions.append(action)
            old_logps.append(logp)
            rewards.append(reward)
            values.append(value.item())
            durations.append(info["dt"])
            if done:
                break
        else:
            raise RuntimeError("episode decision budget exceeded")
        adv, returns = gae(rewards, values, durations, gamma, lam)
        all_adv.extend(adv)
        all_returns.extend(returns)
        episodes.append(
            {
                "seed": seed,
                "initial_gap": initial,
                "gap": env.landscape.gap(env.position),
                "decisions": len(rewards),
            }
        )
    return {
        "data": stack(observations),
        "critic": torch.cat(critics),
        "actions": torch.cat(actions),
        "logp": torch.cat(old_logps),
        "advantages": torch.tensor(all_adv, device=device),
        "returns": torch.tensor(all_returns, device=device),
    }, episodes


def update(
    model, optimizer, batch, epochs=4, minibatch=128, clip=0.2, entropy_coef=0.01, target_kl=0.03
):
    advantages = batch["advantages"]
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    metrics = []
    for _ in range(epochs):
        epoch_kl = []
        for indices in torch.randperm(len(advantages), device=advantages.device).split(minibatch):
            data = {key: value[indices] for key, value in batch["data"].items()}
            _, logp, entropy = model.act(data, batch["actions"][indices])
            values = model.value(batch["critic"][indices])
            logratio = logp - batch["logp"][indices]
            ratio = logratio.exp()
            adv = advantages[indices]
            actor = -torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
            critic = (values - batch["returns"][indices]).square().mean()
            loss = actor + 0.5 * critic - entropy_coef * entropy.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite PPO loss")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5, error_if_nonfinite=True)
            optimizer.step()
            kl = ((ratio - 1) - logratio).mean().detach().item()
            epoch_kl.append(kl)
            metrics.append([loss.item(), actor.item(), critic.item(), entropy.mean().item(), kl])
        if np.mean(epoch_kl) > target_kl:
            break
    return dict(
        zip(
            ("loss", "actor_loss", "value_loss", "entropy", "approx_kl"),
            np.mean(metrics, axis=0).tolist(),
        )
    )


def imitate(model, optimizer, config, seeds, device, epochs=20):
    """Behavior cloning of SGD's discrete choices and gradient learning rate."""
    observations, targets = [], []
    for seed in seeds:
        env, teacher = SoloEnv(config, seed), DecayingSGD()
        while not env.done:
            observation = env.observe()
            data = pack(observation, model.settings["capacity"], device)
            old_steps = teacher.steps
            action = teacher.act(observation, env.compose_walk)
            target = torch.zeros(1, 8, device=device)
            target[0, 0] = KINDS.index(action.kind)
            target[0, 1] = action.batch - 1
            target[0, 6] = np.log(teacher.learning_rate / (1 + teacher.decay * old_steps))
            # Approximate zero position mixing with a small positive log-normal value.
            target[0, 7] = -8
            if action.kind == "walk":
                target[0, 2] = np.flatnonzero(observation["records"][:, 0] == 1)[-1]
            if action.slots:
                target[0, 4] = action.slots[0]
            observations.append(data)
            targets.append(target)
            env.step(action)
    if not observations:
        return
    data, targets = stack(observations), torch.cat(targets)
    for _ in range(epochs):
        for ids in torch.randperm(len(targets), device=device).split(128):
            _, logp, _ = model.act({k: v[ids] for k, v in data.items()}, targets[ids])
            loss = -logp.mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5, error_if_nonfinite=True)
            optimizer.step()


def evaluate(model, config, seeds, device):
    results = []
    for seed in seeds:
        env = SoloEnv(config, seed)
        counts = dict.fromkeys(KINDS, 0)
        with torch.no_grad():
            for _ in range(100_000):
                data = pack(env.observe(), model.settings["capacity"], device)
                action, _, _ = model.act(data, deterministic=True)
                decoded = decode_action(env, action[0])
                counts[decoded.kind] += 1
                env.step(decoded)
                if env.done:
                    break
            else:
                raise RuntimeError("evaluation decision budget exceeded")
        baseline = rollout(SoloEnv(config, seed), DecayingSGD())
        results.append(
            {
                "seed": seed,
                "ppo_gap": env.landscape.gap(env.position),
                "sgd_gap": baseline["final_gap"],
                "actions": counts,
            }
        )
    return {
        "episodes": results,
        "ppo_median_log10_gap": float(np.median([np.log10(r["ppo_gap"] + 1e-12) for r in results])),
        "sgd_median_log10_gap": float(np.median([np.log10(r["sgd_gap"] + 1e-12) for r in results])),
    }


def save_checkpoint(
    path, model, optimizer, config, iteration, next_seed, training=None, algorithm="ppo", extra=None
):
    torch.save(
        {
            **(extra or {}),
            "format_version": 1,
            "algorithm": algorithm,
            "training": training or {},
            "model_settings": model.settings,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "iteration": iteration,
            "next_seed": next_seed,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        path,
    )


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["format_version"] != 1 or checkpoint.get("algorithm", "ppo") != "ppo":
        raise ValueError("expected a supported solo PPO checkpoint")
    model = ActorCritic(**checkpoint["model_settings"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, Config(**checkpoint["config"]), checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--episodes-per-update", type=int, default=8)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--bc-episodes", type=int, default=8)
    parser.add_argument("--bc-epochs", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--output", type=Path, default=Path("results/ppo"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--evaluate", type=Path, help="evaluate checkpoint without training")
    args = parser.parse_args()
    if (
        min(
            args.iterations, args.episodes_per_update, args.epochs, args.eval_episodes, args.threads
        )
        < 1
        or args.bc_episodes < 0
        or args.bc_epochs < 1
        or not 0 < args.gamma <= 1
        or not 0 <= args.gae_lambda <= 1
        or args.learning_rate <= 0
    ):
        parser.error("invalid training parameters")
    if args.resume and args.evaluate:
        parser.error("choose either resume or evaluate")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = args.device
    config = Config(dimension=args.dimension, duration=args.duration)
    start_iteration, next_seed = 0, args.seed
    if args.resume or args.evaluate:
        model, config, checkpoint = load_checkpoint(args.resume or args.evaluate, device)
        start_iteration, next_seed = checkpoint["iteration"], checkpoint["next_seed"]
    else:
        model = ActorCritic(config.notebook_size, config.batch_max).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer"])
        for key, value in checkpoint.get("training", {}).items():
            setattr(args, key, value)
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if device == "cuda" and checkpoint["cuda_rng"]:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.resume and not args.evaluate:
        imitate(
            model,
            optimizer,
            config,
            range(next_seed, next_seed + args.bc_episodes),
            device,
            epochs=args.bc_epochs,
        )
        next_seed += args.bc_episodes
    if not args.evaluate:
        for iteration in range(start_iteration + 1, start_iteration + args.iterations + 1):
            seeds = range(next_seed, next_seed + args.episodes_per_update)
            batch, episodes = collect(model, config, seeds, device, args.gamma, args.gae_lambda)
            next_seed += args.episodes_per_update
            metrics = update(model, optimizer, batch, epochs=args.epochs)
            entry = {
                "iteration": iteration,
                "transitions": len(batch["actions"]),
                "median_log10_gap": float(
                    np.median([np.log10(e["gap"] + 1e-12) for e in episodes])
                ),
                **metrics,
            }
            with (args.output / "train.jsonl").open("a") as stream:
                stream.write(json.dumps(entry) + "\n")
            print(json.dumps(entry), flush=True)
            save_checkpoint(
                args.output / "checkpoint.pt",
                model,
                optimizer,
                config,
                iteration,
                next_seed,
                training={
                    "gamma": args.gamma,
                    "gae_lambda": args.gae_lambda,
                    "epochs": args.epochs,
                },
            )
    # Holdout seeds occupy a separate range; SGD gets the exact same worlds.
    result = evaluate(
        model,
        config,
        range(1_000_000 + args.seed, 1_000_000 + args.seed + args.eval_episodes),
        device,
    )
    (args.output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "run.json").write_text(
        json.dumps(
            {
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "environment": asdict(config),
                "model": model.settings,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
