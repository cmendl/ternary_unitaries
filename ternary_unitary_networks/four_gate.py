"""
Basic functions for 4-particle gates and specific functions to create one via
4-gate construction (not neccesarily unitary in any way).
"""
import numpy as np


def construct_four_gate(p, q, u, v, d=2):
    """
    Creates a 4-qudit gate from the four 2-qudit gates p, q, u and v via the
    four gate construction.
    """
    p = np.reshape(p, (d, d, d, d))
    q = np.reshape(q, (d, d, d, d))
    r = np.reshape(
            np.einsum(p, (0, 2, 4, 6), q, (1, 3, 5, 7), (0, 1, 2, 3, 4, 5, 6, 7)),
            (d**4, d**4))
    return r @ np.kron(u, v)


def four_gate_transpose(u, axis, d=2):
    """
    Reordering dimensions for view along one of the spatial directions.
    """
    u = np.reshape(u, (d, d, d, d, d, d, d, d))
    if axis == 0:
        # apply matrix along x-direction (instead t-direction)
        # by interchanging legs 1 <-> 4 and 3 <-> 6
        u = np.transpose(u, (0, 4, 2, 6, 1, 5, 3, 7))
    elif axis == 1:
        # apply matrix along y-direction (instead t-direction)
        # by interchanging legs 2 <-> 4 and 3 <-> 5
        u = np.transpose(u, (0, 1, 4, 5, 2, 3, 6, 7))
    else:
        assert False, f"invalid argument 'axis = {axis}', must be 0 or 1"
    u = np.reshape(u, (d**4, d**4))
    return u


def conjugation_trace_map_4sites(u, axis, d=2):
    """
    Unitary conjugation and partial trace map for
    calculating correlation functions on a two-dimensional lattice.
    """
    assert u.shape == (d**4, d**4)
    u = np.reshape(u, (d, d, d, d, d, d, d, d))
    if axis == 0:
        m = np.einsum(u, (8, 1, 2, 3, 4, 5, 6, 10), u.conj(), (9, 1, 2, 3, 4, 5, 6, 11), (11, 10, 9, 8)) / d**3
        return np.reshape(m, (d**2, d**2))
    elif axis == 1:
        m = np.einsum(u, (0, 8, 2, 3, 4, 5, 10, 7), u.conj(), (0, 9, 2, 3, 4, 5, 11, 7), (11, 10, 9, 8)) / d**3
        return np.reshape(m, (d**2, d**2))
    elif axis == 2:
        m = np.einsum(u, (0, 1, 8, 3, 4, 10, 6, 7), u.conj(), (0, 1, 9, 3, 4, 11, 6, 7), (11, 10, 9, 8)) / d**3
        return np.reshape(m, (d**2, d**2))
    elif axis == 3:
        m = np.einsum(u, (0, 1, 2, 8, 10, 5, 6, 7), u.conj(), (0, 1, 2, 9, 11, 5, 6, 7), (11, 10, 9, 8)) / d**3
        return np.reshape(m, (d**2, d**2))
    elif axis == (0, 1):
        m = np.einsum(u, (10, 11, 2, 3, 4, 5, 14, 15), u.conj(), (8, 9, 2, 3, 4, 5, 12, 13), (12, 13, 14, 15, 8, 9, 10, 11)) / d**2
        return np.reshape(m, (d**4, d**4))
    elif axis == (0, 2):
        m = np.einsum(u, (10, 1, 11, 3, 4, 14, 6, 15), u.conj(), (8, 1, 9, 3, 4, 12, 6, 13), (12, 13, 14, 15, 8, 9, 10, 11)) / d**2
        return np.reshape(m, (d**4, d**4))
    elif axis == (1, 3):
        m = np.einsum(u, (0, 10, 2, 11, 14, 5, 15, 7), u.conj(), (0, 8, 2, 9, 12, 5, 13, 7), (12, 13, 14, 15, 8, 9, 10, 11)) / d**2
        return np.reshape(m, (d**4, d**4))
    elif axis == (2, 3):
        m = np.einsum(u, (0, 1, 10, 11, 14, 15, 6, 7), u.conj(), (0, 1, 8, 9, 12, 13, 6, 7), (12, 13, 14, 15, 8, 9, 10, 11)) / d**2
        return np.reshape(m, (d**4, d**4))
    else:
        assert False, f"invalid argument axis = {axis}, should be integer or integer tuple"
