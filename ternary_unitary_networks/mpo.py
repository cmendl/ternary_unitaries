import numpy as np
from .bond_ops import split_matrix_svd


class MPO(object):
    """
    Matrix product operator (MPO) class.

    The i-th MPO tensor has dimension `[d, d, D[i], D[i+1]]` with `d` the physical
    dimension at each site and `D` the list of virtual bond dimensions.
    """
    def __init__(self, Alist):
        """
        Create a matrix product operator.
        """
        self.A = [np.array(Aj, copy=False) for Aj in Alist]
        # consistency checks
        for i in range(len(self.A)-1):
            assert self.A[i].ndim == 4
            assert self.A[i].shape[3] == self.A[i+1].shape[2]
        assert self.A[0].shape[2] == self.A[-1].shape[3]

    @classmethod
    def identity(cls, d, L, dtype=complex):
        """
        Construct MPO representation of the identity operation.
        """
        return cls([np.identity(d, dtype=dtype).reshape((d, d, 1, 1)) for _ in range(L)])

    @property
    def nsites(self):
        """
        Number of lattice sites.
        """
        return len(self.A)

    @property
    def bond_dims(self):
        """
        Virtual bond dimensions.
        """
        if len(self.A) == 0:
            return []
        D = [self.A[i].shape[2] for i in range(len(self.A))]
        D.append(self.A[-1].shape[3])
        return D

    def as_matrix(self):
        """
        Merge all tensors to obtain the matrix representation on the full Hilbert space.
        """
        op = self.A[0]
        for i in range(1, len(self.A)):
            op = merge_mpo_tensor_pair(op, self.A[i])
        assert op.ndim == 4
        # contract leftmost and rightmost virtual bond (has no influence if these virtual bond dimensions are 1)
        op = np.trace(op, axis1=2, axis2=3)
        return op

    def canonicalize(self, mode="left"):
        """
        Left- or right-canonicalize the MPO using QR decompositions.
        Note that trailing or leading tensor will not be in canonical form.
        """
        if mode == "left":
            for i in range(len(self.A) - 1):
                # perform QR decomposition and replace A[i] by reshaped Q matrix
                s = self.A[i].shape
                Q, R = np.linalg.qr(self.A[i].reshape((s[0]*s[1]*s[2], s[3])), mode="reduced")
                self.A[i] = Q.reshape((s[0], s[1], s[2], Q.shape[1]))
                # update next tensor: multiply with R from left
                self.A[i+1] = np.einsum(self.A[i+1], (0, 1, 2, 4), R, (3, 2), (0, 1, 3, 4))
        elif mode == "right":
            for i in reversed(range(1, len(self.A))):
                # flip left and right virtual bond dimensions
                self.A[i] = self.A[i].transpose((0, 1, 3, 2))
                # perform QR decomposition and replace A[i] by reshaped Q matrix
                s = self.A[i].shape
                Q, R = np.linalg.qr(self.A[i].reshape((s[0]*s[1]*s[2], s[3])), mode="reduced")
                self.A[i] = Q.reshape((s[0], s[1], s[2], Q.shape[1])).transpose((0, 1, 3, 2))
                # update previous tensor: multiply with R from right
                self.A[i-1] = np.einsum(self.A[i-1], (0, 1, 2, 3), R, (4, 3), (0, 1, 2, 4))
        else:
            raise ValueError('mode = {} invalid; must be "left" or "right"'.format(mode))
        # return self-reference, to enable chaining
        return self

    def apply_two_site_operator(self, i, op, svd_distr="sqrt", tol=0):
        """
        Apply an operator `op` to two neighboring sites.
        The operator acts on both the input and output dimensions,
        and is represented as matrix.
        """
        assert 0 <= i < self.nsites - 1
        assert op.ndim == 2 and op.shape[0] == op.shape[1]
        # local physical dimensions
        d = [self.A[i].shape[0], self.A[i+1].shape[0]]
        A = merge_mpo_tensor_pair(self.A[i], self.A[i+1])
        # combine input and output physical dimensions into a single dimension,
        # and virtual bond dimensions likewise into a single dimension,
        # such that application of the operator becomes a matrix-matrix multiplication
        s = A.shape
        A = np.reshape(A, (s[0]*s[1], s[2]*s[3]))
        A = np.matmul(op, A)
        A = np.reshape(A, s)
        self.A[i], self.A[i+1] = split_mpo_tensor(A, d, svd_distr, tol)

    def __matmul__(self, other):
        """
        Logical multiplication of two MPOs (composition along physical dimension).
        """
        # number of lattice sites must agree
        assert self.nsites == other.nsites
        Alist = []
        for i in range(self.nsites):
            # multiply physical dimensions and reorder dimensions
            A = np.einsum(self.A[i], (0, 1, 3, 5), other.A[i], (1, 2, 4, 6), (0, 2, 3, 4, 5, 6))
            # merge virtual bonds
            s = A.shape
            A = A.reshape((s[0], s[1], s[2]*s[3], s[4]*s[5]))
            Alist.append(A)
        return MPO(Alist)


def merge_mpo_tensor_pair(A0, A1):
    """
    Merge two neighboring MPO tensors.
    """
    A = np.tensordot(A0, A1, (3, 2))
    # pair original physical dimensions of A0 and A1
    A = A.transpose((0, 3, 1, 4, 2, 5))
    # combine original physical dimensions
    s = A.shape
    A = A.reshape((s[0]*s[1], s[2]*s[3], s[4], s[5]))
    return A


def split_mpo_tensor(A, d, svd_distr, tol=0):
    """
    Split a MPO tensor with dimension `d[0]*d[1] x d[0]*d[1] x D0 x D2` into two MPO tensors
    with dimensions `d[0] x d[0] x D0 x D1` and `d[1] x d[1] x D1 x D2`, respectively.
    """
    assert A.ndim == 4
    assert d[0] * d[1] == A.shape[0] == A.shape[1], 'physical dimension of MPO tensor must be equal to d[0] * d[1]'
    # reshape as matrix and split by SVD
    A = A.reshape((d[0], d[1], d[0], d[1], A.shape[2], A.shape[3])).transpose((0, 2, 4, 1, 3, 5))
    s = A.shape
    A0, sigma, A1 = split_matrix_svd(A.reshape((s[0]*s[1]*s[2], s[3]*s[4]*s[5])), tol)
    # use broadcasting to distribute singular values
    if svd_distr == "left":
        A0 = A0 * sigma
    elif svd_distr == "right":
        A1 = A1 * sigma[:, None]
    elif svd_distr == "sqrt":
        A0 = A0 * np.sqrt(sigma)
        A1 = A1 * np.sqrt(sigma)[:, None]
    else:
        raise ValueError('svd_distr parameter must be "left", "right" or "sqrt".')
    A0 = A0.reshape((s[0], s[1], s[2], len(sigma)))
    A1 = A1.reshape((len(sigma), s[3], s[4], s[5]))
    # move physical dimensions to the front
    A1 = A1.transpose((1, 2, 0, 3))
    return A0, A1
