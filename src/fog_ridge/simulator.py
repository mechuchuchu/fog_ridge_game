"""Solo reference simulator. step() advances to the next decision, not a fixed tick."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .landscape import Landscape, Vector


@dataclass(frozen=True)
class Config:
    dimension: int = 2
    duration: float = 600.0
    measurement_time: float = 1.0
    latency: float = 1.0
    batch_max: int = 8
    notebook_size: int = 20
    dashboard_period: float = 5.0
    condition: float = 10.0
    spread: float = 1.0
    start_radius: float = 10.0
    step_max: float | None = None

    def __post_init__(self):
        for name in ("dimension", "batch_max", "notebook_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.batch_max > 100:
            raise ValueError("batch_max cannot exceed map count")
        for name in (
            "duration",
            "measurement_time",
            "latency",
            "dashboard_period",
            "condition",
            "spread",
            "start_radius",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.condition < 1:
            raise ValueError("condition must be >= 1")
        if self.step_max is not None and (not np.isfinite(self.step_max) or self.step_max <= 0):
            raise ValueError("step_max must be finite and positive")


@dataclass(frozen=True)
class Action:
    kind: Literal["measure", "score", "walk", "wait", "delete", "summarize"]
    batch: int = 1
    displacement: Vector | None = None
    seconds: float = 1.0
    slots: tuple[int, ...] = ()


@dataclass
class Record:
    kind: str
    position: Vector
    time: float
    direction: Vector | None = None
    value: float | None = None
    batch: int = 0


class SoloEnv:
    def __init__(self, config=None, seed=0):
        config = Config() if config is None else config
        self.config = config
        # Separate world and sampling streams so policy choices never change the world.
        world, samples = np.random.SeedSequence(seed).spawn(2)
        rng = np.random.default_rng(world)
        self.rng = np.random.default_rng(samples)
        self.landscape = Landscape.generate(rng, config.dimension, config.condition, config.spread)
        offset = rng.normal(size=config.dimension)
        self.position = (
            self.landscape.center + offset / np.linalg.norm(offset) * config.start_radius
        )
        self.step_max = config.step_max or config.start_radius / 3
        self.current_map = int(self.rng.integers(100))
        self.time = 0.0
        self.done = False
        self.records: list[Record] = []
        self.pending: Record | None = None
        self.last_measurement = None
        self.dashboard = None
        self.dashboard_time = 0.0
        self.next_dashboard = config.dashboard_period
        self.trace = []

    @property
    def potential(self):
        return -np.log10(self.landscape.gap(self.position) + 1e-12)

    def action_mask(self):
        kinds = ("measure", "score", "walk", "wait", "delete", "summarize")
        if self.done:
            return dict.fromkeys(kinds, False)
        can_summarize = sum(r.direction is not None for r in self.records) >= 2
        if self.pending is not None:
            return {k: k == "delete" or (k == "summarize" and can_summarize) for k in kinds}
        return {
            k: (
                can_summarize if k == "summarize" else bool(self.records) if k == "delete" else True
            )
            for k in kinds
        }

    def _insert(self, record):
        if len(self.records) == self.config.notebook_size:
            self.pending = record
        else:
            self.records.append(record)

    def _flush_pending(self):
        if self.pending is not None:
            self.records.append(self.pending)
            self.pending = None

    def _vector(self, vector):
        result = np.asarray(vector, dtype=float)
        if result.shape != (self.config.dimension,) or not np.isfinite(result).all():
            raise ValueError("displacement must be a finite D-dimensional vector")
        return result.copy()

    def step(self, action: Action):
        if not self.action_mask().get(action.kind, False):
            raise ValueError(f"illegal action: {action.kind}")
        before = self.potential
        start = self.time
        if action.kind in ("delete", "summarize"):
            count = 1 if action.kind == "delete" else 2
            if (
                len(action.slots) != count
                or len(set(action.slots)) != count
                or any(
                    not isinstance(i, int) or i < 0 or i >= len(self.records) for i in action.slots
                )
            ):
                raise ValueError("invalid notebook slots")
            if count == 2:
                a, b = (self.records[i] for i in action.slots)
                if a.direction is None or b.direction is None:
                    raise ValueError("only direction records can be summarized")
                merged = Record(
                    "summary",
                    (a.position + b.position) / 2,
                    max(a.time, b.time),
                    (a.direction + b.direction) / 2,
                )
            for i in sorted(action.slots, reverse=True):
                del self.records[i]
            if count == 2:
                self.records.append(merged)
            self._flush_pending()
            return self.observe(), 0.0, False, {"dt": 0.0}

        displacement = None
        if action.kind == "measure":
            if not isinstance(action.batch, int) or not 1 <= action.batch <= self.config.batch_max:
                raise ValueError("invalid batch size")
            duration = action.batch * self.config.measurement_time
        elif action.kind == "score":
            duration = 3 * self.config.measurement_time
        elif action.kind == "walk":
            displacement = self._vector(action.displacement)
            norm = np.linalg.norm(displacement)
            if norm > self.step_max:
                displacement *= self.step_max / norm
            duration = self.config.measurement_time / 10
        else:
            if not np.isfinite(action.seconds) or action.seconds <= 0:
                raise ValueError("wait duration must be finite and positive")
            duration = action.seconds

        finish = start + duration
        end = min(finish, self.config.duration)
        # Dashboard ticks strictly before completion see the old measurement.
        while self.next_dashboard < end:
            self._publish()
        self.time = end
        completed = finish <= self.config.duration
        if completed:
            if action.kind == "walk":
                self.position += displacement
                self.current_map = int(self.rng.integers(100))
            elif action.kind == "measure":
                others = np.delete(np.arange(100), self.current_map)
                indices = np.r_[
                    self.current_map, self.rng.choice(others, action.batch - 1, replace=False)
                ]
                value, direction = self.landscape.measure(self.position, indices)
                self.last_measurement = value
                self._insert(
                    Record("measure", self.position.copy(), end, direction, value, action.batch)
                )
            elif action.kind == "score":
                self._insert(
                    Record(
                        "score",
                        self.position.copy(),
                        end,
                        value=self.landscape.score(self.position),
                    )
                )
        if self.next_dashboard == end:
            self._publish()
        self.done = end >= self.config.duration
        reward = float(self.potential - before)
        info = {
            "dt": end - start,
            "completed": completed,
            "gap": self.landscape.gap(self.position),
            "score": self.landscape.score(self.position),
        }
        self.trace.append(
            {"time": end, "action": action.kind, "batch": action.batch, "reward": reward, **info}
        )
        return self.observe(), reward, self.done, info

    def _publish(self):
        self.dashboard = self.last_measurement
        self.dashboard_time = self.next_dashboard
        self.next_dashboard += self.config.dashboard_period

    def observe(self):
        """Actor-safe, dimension-independent scalar features. No true score or axes."""
        n = len(self.records)
        directions = np.zeros((n, self.config.dimension))
        offsets = np.zeros_like(directions)
        tokens = []
        for i, record in enumerate(self.records):
            offset = record.position - self.position
            direction = (
                np.zeros(self.config.dimension) if record.direction is None else record.direction
            )
            directions[i] = direction
            offsets[i] = offset
            dn, pn = np.linalg.norm(direction), np.linalg.norm(offset)
            tokens.append(
                [
                    record.kind == "measure",
                    record.kind == "score",
                    record.kind == "summary",
                    self.time - record.time,
                    record.batch,
                    dn,
                    pn,
                    float(direction @ offset) / max(dn * pn, 1e-12),
                    record.value if record.kind == "score" else 0.0,
                ]
            )
        norms = np.linalg.norm(directions, axis=1)
        cosines = directions @ directions.T / np.maximum(norms[:, None] * norms[None, :], 1e-12)
        offset_norms = np.linalg.norm(offsets, axis=1)
        cross = directions @ offsets.T / np.maximum(norms[:, None] * offset_norms[None, :], 1e-12)
        return {
            "remaining": (self.config.duration - self.time) / self.config.duration,
            "time": self.time,
            "m": self.config.measurement_time,
            "L": self.config.latency,
            "step_max": self.step_max,
            "dashboard": self.dashboard,
            "dashboard_age": self.time - self.dashboard_time,
            "records": np.asarray(tokens, dtype=float).reshape(n, 9),
            "direction_cosines": cosines,
            "direction_offset_cosines": cross,
            "free_slots": self.config.notebook_size - n,
            "mask": self.action_mask(),
        }

    def compose_walk(
        self, gradient_weights, position_weights, gradient_scale=1.0, position_scale=1.0
    ):
        """Decode scalar policy outputs into a world-space action inside the environment."""
        n = len(self.records)
        gw, pw = np.asarray(gradient_weights), np.asarray(position_weights)
        if gw.shape != (n,) or pw.shape != (n,):
            raise ValueError("one weight per record required")
        vector = np.zeros(self.config.dimension)
        for g, p, record in zip(gw, pw, self.records):
            if record.direction is not None:
                vector += gradient_scale * g * record.direction
            vector += position_scale * p * (record.position - self.position)
        return Action("walk", displacement=vector)
