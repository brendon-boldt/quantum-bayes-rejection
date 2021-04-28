from collections import defaultdict
from typing import Mapping

import qiskit as qk  # type: ignore
from qiskit.visualization import *  # type: ignore
import numpy as np

from .network import Network, powerset, get_parents


def prob_to_ang(p: float) -> float:
    """Calculate the angle corresponding to a given probability of 1."""
    return np.arcsin(np.sqrt(p)) * 2


def q_sample_prep(net: Network, inv: bool = False) -> qk.circuit.Gate:
    qreg = qk.QuantumRegister(len(net), "q")
    circ = qk.QuantumCircuit(qreg)

    # The first n bits are the ones we want to sample
    for i, node in enumerate(net):
        all_parents = get_parents(node)
        for parents in powerset(all_parents):
            prob = node.get(parents, 0.0)
            angle = prob_to_ang(prob)
            # We need to undo the rotations of subconditions; e.g., (0, 2) will
            # need to undo the rotations performed by (0,), (2,), and () since
            # those will also be applied if 0 and 2 are true.
            for m, q in node.items():
                if set(m) < set(parents):
                    angle -= prob_to_ang(q)
            if angle == 0.0:
                continue
            if len(parents) == 0:
                circ.ry(angle, qreg[i])
            else:
                circ.mcry(angle, [qreg[idx] for idx in parents], qreg[i])
    gate = circ.to_gate()
    if inv:
        gate = gate.inverse()
        gate.name = "$A_p^\dagger$"
    else:
        gate.name = "$A_p$"
    return gate


def phase_flip(n_qubits: int, evidence: str) -> qk.circuit.Gate:
    qreg = qk.QuantumRegister(n_qubits, "q")
    circ = qk.QuantumCircuit(qreg)
    k = len(evidence)
    for i in range(k):
        if evidence[i] == "0":
            circ.x(qreg[i])
    evidence_bits = [
        b for i, b in enumerate(qreg) if i < len(evidence) and evidence[i] in "01"
    ]
    if len(evidence_bits) == 1:
        circ.p(np.pi, evidence_bits[0])
    elif len(evidence_bits) > 1:
        circ.mcp(np.pi, evidence_bits[:-1], evidence_bits[-1])
    for i in range(k):
        if evidence[i] == "0":
            circ.x(qreg[i])
    gate = circ.to_gate()
    gate.name = "PF"
    return gate


def amplification(n_qubits: int) -> qk.circuit.Gate:
    qreg = qk.QuantumRegister(n_qubits, "q")
    circ = qk.QuantumCircuit(qreg)
    circ.x(qreg)
    circ.mcp(np.pi, qreg[:-1], qreg[-1])
    circ.x(qreg)
    gate = circ.to_gate()
    gate.name = "AA"
    return gate


def make_circuit(net: Network, evidence: str, n_grover_iters: int) -> qk.QuantumCircuit:
    """Make circuit for condition rejection sampling.

    Arguments
    ---------
    evidence: str
        String of evidence bits; "-" refers to an unset bit; e.g., "1-0"
        conditions on x_2=1 and x_0=0

    """
    qreg = qk.QuantumRegister(len(net), "q")
    creg = qk.ClassicalRegister(len(net), "c")
    circ = qk.QuantumCircuit(qreg, creg)

    circ.reset(qreg)

    circ.append(q_sample_prep(net), qreg)
    for i in range(n_grover_iters):
        circ.append(phase_flip(len(net), evidence), qreg)
        circ.append(q_sample_prep(net, inv=True), qreg)
        circ.append(amplification(len(net)), qreg)
        circ.append(q_sample_prep(net), qreg)

    for i in range(len(net)):
        circ.measure(i, i)

    return circ


def simulate_network(
    net: Network, evidence: str, n_grover_iters: int, pdf: bool = False
) -> Mapping[int, float]:
    circuit = make_circuit(net, evidence, n_grover_iters)
    backend = qk.Aer.get_backend("aer_simulator")
    shots = 1 << 14
    result = qk.execute(circuit, backend, shots=shots).result()
    to_pdf = lambda v: v / shots if pdf else v
    counts = {int(k, 2): to_pdf(v) for k, v in result.get_counts().items()}
    return defaultdict(lambda: 0.0, counts)
