from typing import List, Tuple, Mapping, Iterable, TypeVar, Iterator, Any
from collections import defaultdict
from itertools import combinations, chain

import qiskit as qk  # type: ignore
from qiskit.visualization import *  # type: ignore
import numpy as np

_T = TypeVar("_T")

Node = Tuple[Tuple[int, ...], Tuple[float, ...]]
Network = List[Node]


def powerset(iterable: Iterable[_T]) -> Iterator[Tuple[_T, ...]]:
    elems = list(iterable)
    return chain.from_iterable(combinations(elems, l) for l in range(len(elems) + 1))


def get_joint_dist(net: Network) -> Mapping[int, float]:
    result = defaultdict(lambda: 0.0)
    for target in powerset(range(len(net))):
        joint_prob = 1.0
        for node_idx in range(len(net)):
            parents = net[node_idx][0]
            filtered_tgt = tuple(x for x in target if x in parents)
            table_idx = sum(1 << i for i, x in enumerate(parents) if x in filtered_tgt)
            prob_true = net[node_idx][1][table_idx]
            # If the current node is 0 in the joint dist we are calculating, we
            # need to convert p(n=1) to p(n=0).
            prob = prob_true if node_idx in target else 1 - prob_true
            joint_prob *= prob
        result[sum(2 ** int(x) for x in list(target))] = joint_prob
    return result


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

    # Phase correction code
    # Using RY with entangled qubits can cause the phase to become pi whereas
    # we want all phases to be 0 in anticipation of phase flip and
    # amplification. Oddly, the incorrect phases do not appear to affect the
    # algorithm
    tol = 1e-6
    backend = qk.Aer.get_backend("statevector_simulator")
    statevector = qk.execute(circ, backend).result().results[0].data.statevector
    for idx, state in enumerate(statevector):
        # If a state has a phase of pi, flip it to 0
        if state < -tol:
            for i in range(len(net)):
                if (~idx) & (1 << i):
                    circ.x(i)
            circ.mcp(np.pi, qreg[:-1], qreg[-1])
            for i in range(len(net)):
                if (~idx) & (1 << i):
                    circ.x(i)

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
    net: Network,
    evidence: str,
    n_grover_iters: int,
) -> Any:
    circuit = make_circuit(net, evidence, n_grover_iters, measure=False)
    backend = qk.Aer.get_backend("statevector_simulator")
    shots = 1 << 14
    result = qk.execute(circuit, backend, shots=shots).result()
    return result


# Sample networks

# The nodes should be topologically sorted
test_networks: Mapping[str, Network] = {
    # A network in the shape of an "X" with the following edges:
    #     0->2, 1->2, 2->3, 2->4
    "X_0": [
        # No parents, so we only have a prior probability
        ((), (0.25,)),
        ((), (0.5,)),
        # 2 parents -> 4 probabilities for (x0=0, x1=1), (x0=1, x1=0),
        # (x0=0, x1=1), and (x0=1, x1=1)
        ((0, 1), (0.2, 0.4, 0.6, 0.8)),
        ((2,), (0.25, 0.75)),
        ((2,), (0, 1)),
    ],
    "0": [
        ((), (0.25,)),
    ],
    "1": [((), (0.25,)), ((0,), (0.25, 0.75))],
    "2": [((), (0.25,)), ((), (0.25,)), ((0, 1), (0, 0, 0, 1))],
    "pf_0": [
        ((), (0.25,)),
        ((), (0.25,)),
        ((0, 1), (1, 0, 0, 0)),
    ],
    "basic_0": [
        ((), (0.25,)),
        ((), (0.25,)),
        ((0, 1), (1, 0.25, 0, 0.5)),
        ((0, 1, 2), (0, 0, 0, 0, 0, 0, 0, 0.5)),
    ],
}
