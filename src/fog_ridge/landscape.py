"""Quadratic map ensemble; privileged quantities stay inside the simulator."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]


@dataclass
class Landscape:
    centers: Vector
    curvature: Vector

    @classmethod
    def generate(cls, rng, dimension=2, condition=10.0, spread=1.0):
        centers = rng.normal(0, spread / np.sqrt(dimension), (100, dimension))
        curvature = np.exp(rng.uniform(0, np.log(condition), dimension))
        return cls(centers, curvature)

    @property
    def center(self):
        return self.centers.mean(axis=0)

    @property
    def floor(self):
        return float((0.5 * (self.centers - self.center) ** 2 * self.curvature).sum(1).mean())

    def values(self, position, indices=None):
        centers = self.centers if indices is None else self.centers[indices]
        delta = position - centers
        return (0.5 * delta**2 * self.curvature).sum(axis=-1)

    def measure(self, position, indices):
        direction = -self.curvature * (position - self.centers[indices].mean(axis=0))
        return float(self.values(position, indices).mean()), direction

    def gap(self, position):
        return float((0.5 * (position - self.center) ** 2 * self.curvature).sum())

    def score(self, position):
        return self.gap(position) + self.floor
