from scipy.stats import unitary_group
from .dual_unitary import random_dual_gate
from .four_gate import construct_four_gate


def random_dual_4gate():
    """
    Creates a random dual unitary 4-particle gate.

    For this we need two dual unitary gates along the desired direction
    and two normal unitary gates along the other direction.
    """
    d = 2

    dual_unitary_a = random_dual_gate()
    dual_unitary_b = random_dual_gate()

    unitary_a = unitary_group.rvs(d**2)
    unitary_b = unitary_group.rvs(d**2)

    return construct_four_gate(unitary_a, unitary_b, dual_unitary_a, dual_unitary_b)
