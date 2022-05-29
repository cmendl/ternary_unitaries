"""
Visualize equal-time correlation functions.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap


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

    seed = 142
    for t in [1, 2]:
        names = [[f"equal_time_correlation_{seed}.hdf5", f"equal_time_correlation_t{t}.pdf"],
                 [f"equal_time_correlation_dual_{seed}.hdf5", f"equal_time_correlation_dual_t{t}.pdf"]]
        for name in names:
            datafile = name[0]
            savefile = name[1]

            # read data from disk
            with h5py.File("data/" + datafile, "r") as f:
                dset_corr = f["corr"]
                corr = dset_corr[()]
                Δxmax = dset_corr.attrs["Δxmax"]
                Δymax = dset_corr.attrs["Δymax"]

            assert np.allclose(np.linalg.norm(corr.imag), 0)
            corr = corr.real

            plt.imshow(np.log10(np.maximum(abs(corr[t, :, :].T), 1e-7)),
                       cmap=create_colormap(),
                       origin="lower", extent=[-0.5, Δxmax+0.5, -Δymax-0.5, Δymax+0.5])
            plt.colorbar(label=r"$\log_{10}\vert E(a_x^{\alpha}, a_y^{\beta}, t) \vert$")
            plt.xlabel(r"$\Delta x_1$")
            plt.ylabel(r"$\Delta x_2$")
            plt.xticks([0, 5, 10])
            plt.yticks([-10, -5, 0, 5, 10])
            plt.savefig("figures/" + savefile, bbox_inches="tight", pad_inches=0.1)
            plt.close()


if __name__ == "__main__":
    main()
