"""Finite observable-history trees: one control per node enforces nonanticipativity."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ScenarioTree:
    parent: list[int]
    probability: list[float]
    stage: list[int]
    labels: list[str] | None = None
    children: list[list[int]] = field(init=False, repr=False)

    def __post_init__(self):
        self.parent = list(self.parent)
        self.probability = list(self.probability)
        self.stage = list(self.stage)
        n = len(self.parent)
        if not n or len(self.probability) != n or len(self.stage) != n:
            raise ValueError("tree vectors must have equal nonzero lengths")
        if any(not isinstance(x,(int,np.integer)) for x in self.parent+self.stage):
            raise ValueError("parent and stage vectors must contain integers")
        if self.parent[0] != -1 or self.stage[0] != 0 or self.probability[0] != 1:
            raise ValueError("root must have parent=-1, stage=0, probability=1")
        self.children = [[] for _ in range(n)]
        for i in range(1, n):
            p = self.parent[i]
            if not isinstance(p, (int, np.integer)) or not 0 <= p < i:
                raise ValueError("each parent must precede its child")
            if self.stage[i] != self.stage[p]+1:
                raise ValueError("child stage must equal parent stage plus one")
            self.children[p].append(i)
        if not np.isfinite(self.probability).all() or any(p <= 0 for p in self.probability):
            raise ValueError("all represented branches must have positive finite probability")
        for ch in self.children:
            if ch and not np.isclose(sum(self.probability[i] for i in ch), 1,
                                     atol=1e-10, rtol=0):
                raise ValueError("sibling conditional probabilities must sum to one")
        if self.labels is None:
            self.labels = [str(i) for i in range(n)]
        if len(self.labels) != n or len(set(self.labels)) != n:
            raise ValueError("node labels must be unique and complete")

    @property
    def size(self) -> int:
        return len(self.parent)

    @property
    def leaves(self) -> list[int]:
        return [n for n, ch in enumerate(self.children) if not ch]

    @property
    def unconditional(self) -> np.ndarray:
        p = np.ones(self.size)
        for n in range(1, self.size):
            p[n] = p[self.parent[n]] * self.probability[n]
        return p

    def path(self, n: int) -> list[int]:
        if not 0 <= n < self.size:
            raise IndexError(n)
        result = []
        while n >= 0:
            result.append(n)
            n = self.parent[n]
        return result[::-1]

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("parent", "probability", "stage", "labels")}


def regular_tree(depth: int, probabilities: list[float]) -> ScenarioTree:
    if not isinstance(depth, int) or depth < 0:
        raise ValueError("depth must be a nonnegative integer")
    if not probabilities or any(x <= 0 for x in probabilities) or not np.isclose(sum(probabilities), 1):
        raise ValueError("positive branch probabilities summing to one required")
    parent, p, stage = [-1], [1.0], [0]
    frontier = [0]
    for t in range(1, depth+1):
        nxt = []
        for previous in frontier:
            for probability in probabilities:
                nxt.append(len(parent))
                parent.append(previous); p.append(float(probability)); stage.append(t)
        frontier = nxt
    return ScenarioTree(parent, p, stage)
