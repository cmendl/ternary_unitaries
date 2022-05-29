from .dual_unitary import swap_gate, random_dual_gate
from .four_gate import construct_four_gate


def ternary_swap_gate(d=2):
    """
    Construct the ternary-SWAP gate.
    """
    swap = swap_gate(d)
    return construct_four_gate(swap, swap, swap, swap, d)


def construct_random_ternary_gate():
    """
    Construct a random ternary qubit gate from four random dual-unitary gates.
    """
    p = random_dual_gate()
    q = random_dual_gate()
    u = random_dual_gate()
    v = random_dual_gate()
    return construct_four_gate(p, q, u, v, 2)


def dressed_ternary_swap(D, d=2):
    """
    Uses a diagonal unitary matrix D and returns the dressed SWAP gate D*ternary_swap_gate.
    """
    return D @ ternary_swap_gate(d)
