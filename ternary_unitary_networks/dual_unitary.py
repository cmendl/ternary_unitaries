import numpy as np
from scipy.stats import unitary_group


def v_gate(J):
    """
    Creates the gate V[J] usually used in the parametrisation of qudit dual unitaries
    see Eq. (24) of PRL 123, 210601 (2019).
    """
    return np.array([
        [np.exp(-1j*J), 0,               0,               0            ],
        [0,             0,              -1j*np.exp(1j*J), 0            ],
        [0,            -1j*np.exp(1j*J), 0,               0            ],
        [0,             0,               0,               np.exp(-1j*J)]])


def dual_qubit_param(J, a=np.identity(2), b=np.identity(2), c=np.identity(2), d=np.identity(2)):
    """
    Creates a dual unitary gate from the known parametrisation for qubits;
    J is a real number, and u1, v1, u2, v2 are single site unitaries.
    """
    return np.kron(a, b) @ v_gate(J) @ np.kron(c, d)


def random_dual_gate():
    """
    Generate a random dual-unitary gate.
    """
    # random single-qubit gates
    rand_unit_gate_a = unitary_group.rvs(2)
    rand_unit_gate_b = unitary_group.rvs(2)
    rand_unit_gate_c = unitary_group.rvs(2)
    rand_unit_gate_d = unitary_group.rvs(2)
    # random J parameter
    J = 2*np.pi*np.random.uniform()
    return dual_qubit_param(J, rand_unit_gate_a, rand_unit_gate_b, rand_unit_gate_c, rand_unit_gate_d)


def swap_gate(d=2):
    """
    Return the standard SWAP gate (which has the dual unitary property).
    """
    swap = np.identity(d**2)
    swap = np.reshape(swap, (d, d, d, d))
    swap = np.transpose(swap, (0, 1, 3, 2))
    swap = np.reshape(swap, (d**2, d**2))
    return swap
