from typing import Dict, List, Tuple, Any
from functools import reduce

Node = Dict[Tuple[int, ...], float]
Network = List[Node]


def powerset(n: int) -> Any:
    for i in range(1 << n):
        yield tuple(j for j in range(n) if (1 << j) & i)


def get_joint_dist(net: Network) -> List[float]:
    result = []
    parentss = [set(x for k in node.keys() for x in k) for node in net]
    for target in powerset(len(net)):
        joint_prob = 1.0
        print(target)
        for node_idx, parents in enumerate(parentss):
            filtered_tgt = tuple(x for x in target if x in parents)
            prob_true = net[node_idx].get(filtered_tgt, 0.0)
            prob = prob_true if node_idx in target else 1 - prob_true
            print(prob)
            joint_prob *= prob
        result.append(joint_prob)
        print()
    return result


# Sample networks

# The nodes should be topologically sorted, and the node indices should be
# sorted.
simple_network: Network = [
    {(): 0.25},
    {(): 0.25},
    {(0, 1): 0.5, (0,): 0.25, (): 1},
    {(0, 1, 2): 0.5},
]

if __name__ == "__main__":
    probs = get_joint_dist(simple_network)
    for i, p in enumerate(probs):
        print(f"{i:4b} {p:.3f}")
    print(sum(probs))
