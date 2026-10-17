"""Reproducible solo baseline evaluation and optional JSONL event export."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .baseline import DecayingSGD, rollout
from .simulator import Config, SoloEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--condition", type=float, default=10)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--decay", type=float, default=0.02)
    parser.add_argument("--output", type=Path, help="directory for summary.json and events.jsonl")
    args = parser.parse_args()
    if args.episodes < 1 or not 1 <= args.batch <= 8:
        parser.error("episodes must be positive and batch must be in 1..8")
    config = Config(dimension=args.dimension, duration=args.duration, condition=args.condition)
    results, events = [], []
    for seed in range(args.seed, args.seed + args.episodes):
        env = SoloEnv(config, seed)
        policy = DecayingSGD(args.learning_rate, args.decay, args.batch)
        result = rollout(env, policy)
        results.append({"seed": seed, **result})
        events.extend({"seed": seed, **event} for event in env.trace)
    logs = np.log10([r["final_gap"] + 1e-12 for r in results])
    summary = {
        "config": vars(args) | {"output": str(args.output) if args.output else None},
        "log10_gap_quartiles": np.quantile(logs, [0.25, 0.5, 0.75]).tolist(),
        "median_gap_reduction": float(
            np.median([r["initial_gap"] / max(r["final_gap"], 1e-12) for r in results])
        ),
        "actions": dict(Counter(e["action"] for e in events)),
        "episodes": results,
    }
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        with (args.output / "events.jsonl").open("w") as stream:
            for event in events:
                stream.write(json.dumps(event) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
