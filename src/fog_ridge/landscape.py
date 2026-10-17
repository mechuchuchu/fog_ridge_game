"""Quadratic map ensemble; privileged quantities stay inside the simulator."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]


@dataclass
class Landscape:
    centers: Vector
    curvature: Vector
    map_curvature: Vector | None = None
    quartic_strength: float = 0.0
    quartic_scale: float = 1.0

    @classmethod
    def generate(
        cls,
        rng,
        dimension=2,
        condition=10.0,
        spread=1.0,
        *,
        curvature_jitter=0.0,
        quartic_strength=0.0,
        quartic_scale=1.0,
    ):
        curvature = np.exp(rng.uniform(0, np.log(condition), dimension))
        if curvature_jitter or quartic_strength:
            # Paired maps keep the ensemble minimum at the mean center while
            # allowing each map to have its own local curvature and valley.
            offsets = rng.normal(0, spread / np.sqrt(dimension), (50, dimension))
            centers = np.empty((100, dimension))
            centers[0::2], centers[1::2] = offsets, -offsets
            map_curvature = np.empty_like(centers)
            pair_curvature = curvature * np.exp(rng.normal(0, curvature_jitter, (50, dimension)))
            map_curvature[0::2] = map_curvature[1::2] = pair_curvature
        else:
            centers = rng.normal(0, spread / np.sqrt(dimension), (100, dimension))
            map_curvature = None
        return cls(centers, curvature, map_curvature, quartic_strength, quartic_scale)

    def _curvatures(self, indices=None):
        if self.map_curvature is None:
            count = 100 if indices is None else len(np.atleast_1d(indices))
            return np.broadcast_to(self.curvature, (count, len(self.curvature)))
        return self.map_curvature if indices is None else self.map_curvature[indices]

    @property
    def center(self):
        return self.centers.mean(axis=0)

    @property
    def floor(self):
        if self.map_curvature is not None or self.quartic_strength:
            return float(self.values(self.center).mean())
        return float((0.5 * (self.centers - self.center) ** 2 * self.curvature).sum(1).mean())

    def values(self, position, indices=None):
        centers = self.centers if indices is None else self.centers[indices]
        delta = position - centers
        value = (0.5 * delta**2 * self._curvatures(indices)).sum(axis=-1)
        if self.quartic_strength:
            radius_squared = (delta**2).sum(axis=-1)
            value = value + (self.quartic_strength / (4 * self.quartic_scale**2)) * (
                radius_squared**2
            )
        return value

    def measure(self, position, indices):
        delta = position - self.centers[indices]
        curvatures = self._curvatures(indices)
        directions = -curvatures * delta
        if self.quartic_strength:
            radius_squared = (delta**2).sum(axis=-1, keepdims=True)
            directions -= (self.quartic_strength / self.quartic_scale**2) * radius_squared * delta
        return float(self.values(position, indices).mean()), directions.mean(axis=0)

    def gap(self, position):
        if self.map_curvature is not None or self.quartic_strength:
            return max(0.0, self.score(position) - self.floor)
        return float((0.5 * (position - self.center) ** 2 * self.curvature).sum())

    def score(self, position):
        return float(self.values(position).mean())
