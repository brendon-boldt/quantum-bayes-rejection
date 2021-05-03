"""Unit tests for circuit builder"""

import unittest
from typing import List, Tuple

from scipy import stats  # type: ignore

from .. import network
from .. import builder

THRESHOLD = 0.01


def failures_to_string(name: str, fs: List[Tuple[int, float, float, float]]) -> str:
    res = f"{name}: Bad probabilities (key, exact, simulation, significance):"
    for k, calc, sim, sig in fs:
        res += f"\n\t{k:04b}  {calc:.3f} {sim:.3f} {sig:.3f}"
    return res


class CircuitJointDist(unittest.TestCase):
    def _test_joint(self, name: str, net: network.Network) -> None:
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
        explanation = failures_to_string(name, failures)
        self.assertTrue(len(failures) == 0, explanation)

    def test_joint(self) -> None:
        for name, net in network.test_networks.items():
            self._test_joint(name, net)
