"""Unit tests for circuit builder"""

import unittest
from typing import List, Tuple

from scipy import stats  # type: ignore

from .. import network
from .. import builder

THRESHOLD = 0.01


def failures_to_string(fs: List[Tuple[int, float, float, float]]) -> str:
    res = "Bad probabilities (key, exact, simulation, significance):"
    for k, calc, sim, sig in fs:
        res += f"\n\t{k:04b}  {calc:.3f} {sim:.3f} {sig:.3f}"
    return res


class CircuitJoinDist(unittest.TestCase):
    def _test_joint(self, net: network.Network) -> None:
        res_sim = builder.simulate_network(net, "", 0)
        res_calc = network.get_joint_dist(net)
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

    def test_simple_joint(self) -> None:
        self._test_joint(network.simple_network)

    def test_paper_joint(self) -> None:
        self._test_joint(network.paper_network)
