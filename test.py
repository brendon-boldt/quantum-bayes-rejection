import unittest
from typing import List, Tuple

from scipy import stats  # type: ignore

import qbr

THRESHOLD = 0.01


def failures_to_string(name: str, fs: List[Tuple[int, float, float, float]]) -> str:
    res = f"{name}: Bad probabilities (key, exact, simulation, significance):"
    for k, calc, sim, sig in fs:
        res += f"\n\t{k:04b}  {calc:.3f} {sim:.3f} {sig:.3f}"
    return res


class CircuitJointDist(unittest.TestCase):
    def _test_joint(self, name: str, net: qbr.Network) -> None:
        res_sim = qbr.simulate_network(net, "", 0)
        res_calc = qbr.get_joint_dist(net)
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
        for name, net in qbr.test_networks.items():
            self._test_joint(name, net)


def print_hist(d, n_bits, width=60):
    max_val = max(d.values())
    for i in range(2 ** n_bits):
        if d[i] == 0:
            continue
        bit_str = f"{i:b}"
        while len(bit_str) < n_bits:
            bit_str = "0" + bit_str
        print(bit_str, end=" ")
        print(f"{d[i]:.3f} ", end="")
        print("#" * int(d[i] / max_val * width))


def main():
    net = qbr.test_networks["basic_0"]

    res = qbr.simulate_network_statevector(net, "1", 0, pdf=True)
    statevector = res.results[0].data.statevector
    for i in range(2 ** len(net)):
        print(f"{i:04b} {statevector[i]:+.3f}")

    for i in range(3):
        res = builder.simulate_network(net, "10-", i, pdf=True)
        print_hist(res, len(net), 60)
        print()

    # print(builder.make_circuit(net, '01', 0).qasm())


if __name__ == "__main__":
    # Run testing code
    # main()

    # Run tests
    unittest.main()
