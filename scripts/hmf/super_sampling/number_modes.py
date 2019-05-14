import numpy as np
import pynbody
from mlhalos import parameters
import matplotlib.pyplot as plt


def get_k_modes_box(initial_parameters, boxsize=None, units="Mpc**-1 h a**-1"):
    if boxsize is None:
        boxsize = initial_parameters.boxsize_comoving

    a = np.load(initial_parameters.path + "/stored_files/Fourier_transform_matrix.npy")
    a[0,0,0] = 10**9
    k = 2. * np.pi * a / boxsize
    k = pynbody.array.SimArray(k)
    k.units = units
    return k


def sharp_k(k, k_scale):
    modes = np.where(k <= k_scale, 1, 0)
    # return np.where(k == k_scale, 0.5, modes)
    return modes


def get_number_of_modes_below_k_scale(initial_parameters, k_scale):
    ks = get_k_modes_box(initial_parameters)
    assert ks.units == k_scale.units
    if isinstance(k_scale, (int, float)):
        num_k = np.sum(sharp_k(ks, k_scale))
    else:
        num_k = np.array([np.sum(sharp_k(ks, k_i)) for k_i in k_scale])
    return num_k


def analytic_derivation_num_modes_sphere(k_scale, boxsize):
    num_modes = boxsize**3 * k_scale**3 / (6*np.pi**2)
    return num_modes

def analytic_derivation_num_modes_sphere_2(n_k):
    num_modes = (2*np.pi*n_k)**3/ (6*np.pi**2)
    return num_modes


def dn_dk_analytic(k_scale, boxsize):
    dndk = boxsize ** 3 * k_scale ** 2 / (2 * np.pi ** 2)
    return dndk


if __name__ == "__main__":

    ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")
    L = pynbody.array.SimArray(ic.boxsize_comoving)
    L.units = "Mpc h**-1 a"
    delta_k = 2 * np.pi / L
    nyquist = delta_k * ic.shape / 2

    # spacing = np.sqrt(3) / 4 * delta_k
    spacing = delta_k
    k_scales = np.arange(delta_k, nyquist + spacing, spacing).view(pynbody.array.SimArray)
    k_scales.units = delta_k.units

    #k_scales = np.array([delta_k * n /(np.sqrt(3)/2) for n in range(0, 128)])

    num_emp = get_number_of_modes_below_k_scale(ic, k_scales)
    num_th = analytic_derivation_num_modes_sphere(k_scales, ic.boxsize_comoving)

    plt.loglog(k_scales/delta_k, num_emp, label="emp")
    plt.loglog(k_scales/delta_k, num_th, label="analytic")
    # plt.yscale("log")
    plt.xlabel(r"$k/(2\pi/L)$")
    # plt.xlim(0, 10**3)
    plt.ylabel("Number of modes")
    plt.legend(loc="best")
