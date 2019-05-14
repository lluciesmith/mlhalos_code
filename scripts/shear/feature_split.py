import sys
sys.path.append('/Users/lls/Documents/mlhalos_code/scripts')
from mlhalos import shear
from mlhalos import parameters
from scipy.constants import G
import numpy as np
import matplotlib.pyplot as plt


def get_trace_eigenvalues(shear_class, normalise_prefactor=True):
    eig_in = shear_class.shear_eigval_in
    eig_out = shear_class.shear_eigval_out

    trace_in = np.real(np.sum(eig_in, axis=1))
    trace_out = np.real(np.sum(eig_out, axis=1))

    if normalise_prefactor is True:
        trace_in /= (4 * np.pi * G)
        trace_out /= (4 * np.pi * G)

    return trace_in, trace_out


def get_density_minus_mean(shear_class, initial_conditions, ids_type="in"):
    den = shear_class._density_scale
    den_real = np.real(np.fft.ifftn(den).reshape(256 ** 3))

    mean_rho = np.mean(den_real)
    den_diff = den_real - mean_rho

    if ids_type is not None:
        if ids_type == "in":
            ids = initial_conditions.ids_IN
        elif ids_type == "out":
            ids = initial_conditions.ids_OUT
        den_diff_ids = np.array([den_diff[particle_ID] for particle_ID in ids])

    else:
        den_diff_ids = den_diff

    return den_diff_ids


def plot_histogram_density_and_trace(trace_in, trace_out, den_diff_in, den_diff_out, save=True,
                                     path = "/Users/lls/Documents/CODE/stored_files/"
                                            "shear-eigvals/density_27_hist_split.pdf"):
    plt.figure(figsize=(8, 6))
    n, b, p = plt.hist(den_diff_in, label="density-in", histtype='step', color='b', ls='--', lw=2,
                       # normed=True
                       )
    n1, b1, p1 = plt.hist(den_diff_out, label="density-out", histtype='step', color='g', ls='--', lw=2,
                          # normed=True
                          )
    na, ba, pa = plt.hist(trace_in, label="eigvals-in", histtype='step', color='b', ls='-', lw=2, bins=b,
                          # normed=True
                          )
    nb, bb, pb = plt.hist(trace_out, label="eigvals-out", histtype='step', color='g', ls='-', lw=2, bins=b1,
                          # normed=True
                          )
    plt.legend(loc=2)
    plt.xlabel("density - mean density")

    if save is True:
        plt.savefig(path)


if __name__ == "__main__":

    ic = parameters.InitialConditionsParameters()
    s = shear.Shear(initial_parameters=ic, shear_scale=27)

    trace_in, trace_out = get_trace_eigenvalues(s, normalise_prefactor=True)

    den_diff_in = get_density_minus_mean(s, ic, ids_type="in")
    den_diff_out = get_density_minus_mean(s, ic, ids_type="out")

    plot_histogram_density_and_trace(trace_in, trace_out, den_diff_in, den_diff_out, save=True)


