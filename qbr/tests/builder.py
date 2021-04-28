"""Unit tests for circuit builder"""

import unittest
from typing import List, Tuple

from scipy import stats  # type: ignore

from .. import network
from .. import builder

simple_network: network.Network = [
    {(): 0.25},
    {(): 0.25},
    {(0, 1): 0.5, (0,): 0.25, (): 1},
    {(0, 1, 2): 0.5},
]

THRESHOLD = 0.01


def failures_to_string(fs: List[Tuple[int, float, float, float]]) -> str:
    res = "Bad probabilities (key, exact, simulation, significance):"
    for k, calc, sim, sig in fs:
        res += f"\n\t{k:04b}  {calc:.3f} {sim:.3f} {sig:.3f}"
    return res


class CircuitJoinDist(unittest.TestCase):
    def test_simple(self) -> None:
        res_sim = builder.simulate_network(simple_network, "", 0)
        res_calc = network.get_joint_dist(simple_network)
        shots = sum(res_sim.values())
        failures = []
        for k in res_calc:
            sim = res_sim[k]
            calc = res_calc[k]
            sig = stats.binom_test(sim, shots, calc)
            if sig < THRESHOLD:
                failures.append((k, calc, sim / shots, sig))
        explanation = failures_to_string(failures)
        self.assertTrue(len(failures) == 0, explanation)
