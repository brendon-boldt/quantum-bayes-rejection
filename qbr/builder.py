from collections import defaultdict
from typing import Mapping, Any

import qiskit as qk  # type: ignore
from qiskit.visualization import *  # type: ignore
import numpy as np

from .network import Network, powerset


def prob_to_ang(p: float) -> float:
    """Calculate the angle corresponding to a given probability of 1."""
    return np.arcsin(np.sqrt(p)) * 2


def q_sample_prep(net: Network, inv: bool = False) -> qk.circuit.Gate:
    qreg = qk.QuantumRegister(len(net), "q")
    circ = qk.QuantumCircuit(qreg)

    # The first n bits are the ones we want to sample
    for i, node in enumerate(net):
        all_parents = node[0]
        for parents in powerset(all_parents):
            parents_idx = sum(1 << i for i, x in enumerate(all_parents) if x in parents)
            prob = node[1][parents_idx]
            angle = prob_to_ang(prob)
            # We need to undo the rotations of subconditions; e.g., (0, 2) will
            # need to undo the rotations performed by (0,), (2,), and () since
            # those will also be applied if 0 and 2 are true.
            for j, q in enumerate(node[1]):
                # if j is a subset of parents
                if (j & parents_idx) == j and not j == parents_idx:
                    angle -= prob_to_ang(q)
            if np.fmod(angle, 2 * np.pi) == 0.0:
                continue
            if len(parents) == 0:
                circ.ry(angle, qreg[i])
            else:
                circ.mcry(angle, [qreg[idx] for idx in parents], qreg[i])
    gate = circ.to_gate()
    if inv:
        gate = gate.inverse()
        gate.name = "apd"
        gate.label = "$A_p^\dagger$"
    else:
        gate.name = "ap"
        gate.label = "$A_p$"
    return gate


def phase_flip(n_qubits: int, evidence: str) -> qk.circuit.Gate:
    qreg = qk.QuantumRegister(n_qubits, "q")
    circ = qk.QuantumCircuit(qreg)
    # The evidence is indexed backwards, so we have to reverse it.
    r_ev = list(reversed(evidence))
    for i in range(len(r_ev)):
        if r_ev[i] == "0":
            circ.x(qreg[i])
    evidence_bits = [b for i, b in enumerate(qreg) if i < len(r_ev) and r_ev[i] in "01"]
    if len(evidence_bits) == 1:
        circ.p(np.pi, evidence_bits[0])
    elif len(evidence_bits) > 1:
        circ.mcp(np.pi, evidence_bits[:-1], evidence_bits[-1])
    for i in range(len(r_ev)):
        if r_ev[i] == "0":
            circ.x(qreg[i])
    gate = circ.to_gate()
    gate.name = "pf"
    gate.label = "PF"
    return gate


def amplification(n_qubits: int) -> qk.circuit.Gate:
    qreg = qk.QuantumRegister(n_qubits, "q")
    circ = qk.QuantumCircuit(qreg)
    circ.x(qreg)
    circ.mcp(np.pi, qreg[:-1], qreg[-1])
    circ.x(qreg)
    gate = circ.to_gate()
    gate.name = "aa"
    gate.label = "AA"
    return gate


def make_circuit(
    net: Network, evidence: str, n_grover_iters: int, measure: bool = True
) -> qk.QuantumCircuit:
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

    if measure:
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


def simulate_network_statevector(
    net: Network, evidence: str, n_grover_iters: int, pdf: bool = False
) -> Any:
    circuit = make_circuit(net, evidence, n_grover_iters, measure=False)
    backend = qk.Aer.get_backend("statevector_simulator")
    shots = 1 << 14
    result = qk.execute(circuit, backend, shots=shots).result()
    return result
