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
    parser.add_argument("--spread", type=float, default=1)
    parser.add_argument("--start-radius", type=float, default=10)
    parser.add_argument("--step-max", type=float)
    parser.add_argument("--measurement-time", type=float, default=1)
    parser.add_argument("--latency", type=float, default=1)
    parser.add_argument("--batch-max", type=int, default=8)
    parser.add_argument("--notebook-size", type=int, default=20)
    parser.add_argument("--curvature-jitter", type=float, default=0)
    parser.add_argument("--quartic-strength", type=float, default=0)
    parser.add_argument("--no-score", action="store_true", help="disable score actions")
    parser.add_argument(
        "--unit-directions", action="store_true", help="normalize measured directions"
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--decay", type=float, default=0.02)
    parser.add_argument("--output", type=Path, help="directory for summary.json and events.jsonl")
    args = parser.parse_args()
    if args.episodes < 1 or not 1 <= args.batch <= args.batch_max:
        parser.error("episodes must be positive and batch must be in 1..batch-max")
    config = Config(
        dimension=args.dimension,
        duration=args.duration,
        measurement_time=args.measurement_time,
        latency=args.latency,
        batch_max=args.batch_max,
        notebook_size=args.notebook_size,
        condition=args.condition,
        spread=args.spread,
        start_radius=args.start_radius,
        step_max=args.step_max,
        curvature_jitter=args.curvature_jitter,
        quartic_strength=args.quartic_strength,
        score_enabled=not args.no_score,
        unit_directions=args.unit_directions,
    )
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
