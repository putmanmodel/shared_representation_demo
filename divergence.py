import math
from typing import Dict, List, Tuple

from models import Interpretation


Matrix = Dict[str, Dict[str, float]]


def euclidean_distance(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    total = 0.0
    for x, y in zip(a, b):
        total += (x - y) ** 2
    return math.sqrt(total)


def pairwise_divergence_matrix(interpretations: List[Interpretation]) -> Matrix:
    matrix: Matrix = {}
    for left in interpretations:
        matrix[left.agent_id] = {}
        for right in interpretations:
            if left.agent_id == right.agent_id:
                matrix[left.agent_id][right.agent_id] = 0.0
            else:
                matrix[left.agent_id][right.agent_id] = euclidean_distance(
                    left.feature_vector,
                    right.feature_vector,
                )
    return matrix


def mean_pairwise_divergence(matrix: Matrix) -> float:
    ids = list(matrix.keys())
    if len(ids) < 2:
        return 0.0

    total = 0.0
    count = 0
    for i, left_id in enumerate(ids):
        for right_id in ids[i + 1 :]:
            total += matrix[left_id][right_id]
            count += 1
    return total / count if count else 0.0
