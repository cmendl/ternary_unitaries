import numpy as np
from .mpo import merge_mpo_tensor_pair


class iMPO(object):
    """
    Translation-invariant ("infinite") matrix product operator,
    i.e., same tensor `A` at each site.
    `A` has shape (d, d, D, D), with `d` the physical dimension at each site and
    `D` the virtual bond dimension.
    """
    def __init__(self, A):
        """
        Create a translation-invariant matrix product operator.
        """
        A = np.array(A, copy=False)
        assert A.ndim == 4
        assert A.shape[0] == A.shape[1]
        assert A.shape[2] == A.shape[3]
        self.A = A
        self.d = A.shape[0]
        self.D = A.shape[2]

    @classmethod
    def identity(cls, d, dtype=complex):
        """
        Construct iMPO representation of the identity operation.
        """
        return cls(np.identity(d, dtype=dtype).reshape((d, d, 1, 1)))

    def transfer_operator(self):
        """
        Construct the transfer operator.
        """
        return np.einsum(self.A,        (0, 1, 2, 3),
                         self.A.conj(), (0, 1, 4, 5),
                         (2, 4, 3, 5)).reshape((self.D**2, self.D**2)) / self.d

    def block_tensors(self, nsites):
        """
        Construct the blocked MPO tensor by grouping `nsites` physical sites together.
        """
        assert nsites >= 1
        op = self.A
        for i in range(1, nsites):
            op = merge_mpo_tensor_pair(op, self.A)
        return op

    def as_matrix(self, nsites):
        """
        Merge tensors to obtain the matrix representation on the full Hilbert space.
        """
        # contract periodic boundary conditions
        return np.trace(self.block_tensors(nsites), axis1=2, axis2=3)
