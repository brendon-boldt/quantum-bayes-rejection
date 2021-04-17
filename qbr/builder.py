from collections import defaultdict
from typing import Mapping

import qiskit as qk  # type: ignore
from qiskit.visualization import *  # type: ignore
import numpy as np

from .network import Network


def prob_to_ang(p: float) -> float:
    """Calculate the angle corresponding to a given probability of 1."""
    return np.arcsin(np.sqrt(p)) * 2


def make_prep_circuit(net: Network) -> qk.QuantumCircuit:
    qreg = qk.QuantumRegister(len(net), "q")
    creg = qk.ClassicalRegister(len(net), "c")
    circ = qk.QuantumCircuit(qreg, creg)

    circ.reset(qreg)

    # The first n bits are the ones we want to sample
    for i, node in enumerate(net):
        for parents, prob in node.items():
            angle = prob_to_ang(prob)
            # We need to undo the rotations of subconditions; e.g., (0, 2) will
            # need to undo the rotations performed by (0,), (2,), and () since
            # those will also be applied if 0 and 2 are true.
            for m, q in node.items():
                if set(m) < set(parents):
                    angle -= prob_to_ang(q)
            if len(parents) == 0:
                circ.rx(angle, qreg[i])
            else:
                # TODO These might need to be mrcz, not mcry
                # circ.mcrx(angle, [qreg[idx] for idx in parents], qreg[i])
                circ.mcry(angle, [qreg[idx] for idx in parents], qreg[i])
    for i in range(len(net)):
        circ.measure(qreg[i], creg[i])
    return circ


def simulate_network(net: Network, pdf: bool = False) -> Mapping[int, float]:
    circuit = make_prep_circuit(net)
    backend = qk.Aer.get_backend("aer_simulator")
    shots = 1 << 13
    result = qk.execute(circuit, backend, shots=shots).result()
    to_pdf = lambda v: v / shots if pdf else v
    counts = {int(k, 2): to_pdf(v) for k, v in result.get_counts().items()}
    return defaultdict(lambda: 0.0, counts)
