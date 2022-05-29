import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import ternary_unitary_networks as tun


def generate_random_local_operator():
    """
    Sample a Hermitian, traceless single-qubit operator.
    """
    # Pauli matrices
    σ = [np.array([[1,  0 ], [0,  1]]),
         np.array([[0,  1 ], [1,  0]]),
         np.array([[0, -1j], [1j, 0]]),
         np.array([[1,  0 ], [0, -1]])]
    # random Bloch vector
    r = np.random.randn(3)
    r /= np.linalg.norm(r)
    return sum(r[i]*σ[i+1] for i in range(3))


def create_colormap():
    """
    Create custom colormap such that the bottom fades to white.
    """
    viridis = cm.get_cmap("viridis", 256)
    colors = viridis(np.linspace(0, 1, 256))
    # reverse
    colors = colors[::-1]
    white = np.array([1., 1., 1., 1.])
    for i in range(15):
        colors[i, :] = i/15 * colors[i, :] + (1 - i/15)*white
    return ListedColormap(colors)


def main():

    Tmax = 20
    cmap = create_colormap()
    figfiles = ["figures/dynamic_correlation_random.pdf",
                "figures/dynamic_correlation_swap.pdf"]
    for i in range(2):
        np.random.seed(142)
        # ternary unitary gate
        if i == 0:
            G = tun.construct_random_ternary_gate()
        else:
            G = tun.ternary_swap_gate(d=2)
        # local operators
        ax = generate_random_local_operator()
        ay = generate_random_local_operator()
        # compute correlation functions
        corr = tun.compute_dynamic_correlations(G, ax, ay, Tmax, d=2)
        # imaginary part is expected to be zero
        assert np.allclose(np.linalg.norm(corr.imag), 0)
        corr = corr.real
        # convert to diagonal matrix for plotting
        corr = np.diag(corr)
        plt.imshow(np.log10(np.maximum(abs(corr), 1e-7)), origin="lower",
                   cmap=cmap, aspect=2,
                   extent=[-1, 2*Tmax+1, -0.5, Tmax+0.5])
        plt.xlabel(r"$\Vert x - y \Vert_\infty$")
        plt.ylabel(r"$t$")
        plt.colorbar(label=r"$\log_{10} \vert D(a_x^\alpha, a_y^\beta,t) \vert$")
        plt.yticks([0,  5, 10, 15, 20])
        plt.xticks([0, 10, 20, 30, 40])
        plt.show()
        # save figure to disk
        plt.savefig(figfiles[i], bbox_inches="tight", pad_inches=0.1)
        plt.close()


if __name__ == "__main__":
    main()
