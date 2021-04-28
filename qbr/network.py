from typing import Dict, List, Tuple, Mapping, Iterable, TypeVar, Iterator
from collections import defaultdict
from itertools import combinations, chain

_T = TypeVar("_T")

Node = Dict[Tuple[int, ...], float]
Network = List[Node]


def powerset(iterable: Iterable[_T]) -> Iterator[Tuple[_T, ...]]:
    elems = list(iterable)
    return chain.from_iterable(combinations(elems, l) for l in range(len(elems) + 1))


def get_parents(node: Node) -> Tuple[int, ...]:
    return tuple(set(x for k in node.keys() for x in k))


def get_joint_dist(net: Network) -> Mapping[int, float]:
    result = defaultdict(lambda: 0.0)
    for target in powerset(range(len(net))):
        joint_prob = 1.0
        for node_idx in range(len(net)):
            parents = get_parents(net[node_idx])
            filtered_tgt = tuple(x for x in target if x in parents)
            prob_true = net[node_idx].get(filtered_tgt, 0.0)
            # If the current node is 0 in the joint dist we are calculating, we
            # need to convert p(n=1) to p(n=0).
            prob = prob_true if node_idx in target else 1 - prob_true
            joint_prob *= prob
        result[sum(2 ** int(x) for x in list(target))] = joint_prob
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

paper_network: Network = [
    {(): 0.5},
    {(0,): 0.2},
    {(0,): 0.9},
    {(0,): 0.1},
    {(1, 2): 0.3},
    {(3, 4): 0.5},
    {(3,): 0.7},
]
