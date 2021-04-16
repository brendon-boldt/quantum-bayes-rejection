from typing import Dict, List, Tuple

from qiskit import QuantumCircuit  # type: ignore
from qiskit.visualization import *  # type: ignore
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
import numpy as np  # type: ignore

Node = Dict[Tuple[int, ...], float]
Network = List[Node]

# This is assumed to be topologically sorted
net: Network = [
    {(): 0.25},
    {(): 0.25},
    {(0, 1): 0.5, (0,): 0.25, (): 1},
    {(0, 1, 2): 0.5},
]


def prob_to_ang(p: float) -> float:
    """Calculate the angle corresponding to a given probability of 1."""
    return np.arcsin(np.sqrt(p)) * 2


def make_prep_circuit(net: Network) -> QuantumCircuit:
    qreg = QuantumRegister(len(net), "q")
    creg = ClassicalRegister(len(net), "c")
    circ = QuantumCircuit(qreg, creg)

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
