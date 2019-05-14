import sys

import numpy as np
import pynbody

from scripts.ellipsoidal.power_spectrum import get_power_spectrum
camb_path = "/home/lls/stored_files/"
#camb_path = "/Users/lls/Software/CAMB-Jan2017/"
#mlhalos_path = "/Users/lls/Documents/mlhalos_code"
#sys.path.append(mlhalos_path)
sys.path.append(camb_path)
from mlhalos import window
from mlhalos import parameters
import matplotlib.pyplot as plt


def get_variance(smoothing_scales, window_function, power_spectrum, snapshot):
    """ Mass needs to be in (Msol h^-1) units and radius needs to be in (Mpc h^-1 * a) units """
    if smoothing_scales.units == "Msol":
        h = snapshot.properties['h']

        smoothing_scales = smoothing_scales * h
        smoothing_scales.units = "Msol h**-1"

        arg_is_R = False

    elif smoothing_scales.units == "Msol h^-1":
        arg_is_R = False

    elif smoothing_scales.units == "Mpc":
        h = snapshot.properties['h']
        a = snapshot.properties['a']

        smoothing_scales = smoothing_scales * h / a
        smoothing_scales.units = "Mpc h**-1 a"

        arg_is_R = True

    elif smoothing_scales.units == "Mpc h**-1 a":
        arg_is_R = True

    elif smoothing_scales.units == "Mpc**-1 a**-1 h":
        smoothing_scales = 2*np.pi/smoothing_scales
        arg_is_R = True

    else:
        raise NameError("Select either radius/wavenumber or mass smoothing scales")

    var = pynbody.analysis.hmf.variance(smoothing_scales, window_function, power_spectrum, arg_is_R=arg_is_R)
    return var


def calculate_variance_radius(smoothing_scales, ic, z=99, cosmology="WMAP5", filter=None):
    if z == 99:
        snapshot = ic.initial_conditions
    elif z == 0:
        snapshot = ic.final_snapshot
    else:
        raise NameError("Select a valid redshift")

    powerspec = get_power_spectrum(cosmology, ic, z=z)
    print("Done getting power spectrum")
    if filter is None:
        filter = powerspec._default_filter

    var = pynbody.analysis.hmf.variance(smoothing_scales, filter, powerspec, arg_is_R=True)

    return var


def calculate_variance(smoothing_scales, ic, z=99, cosmology="WMAP5", filter=None):
    if z == 99:
        snapshot = ic.initial_conditions
    elif z == 0:
        snapshot = ic.final_snapshot
    else:
        raise NameError("Select a valid redshift")

    powerspec = get_power_spectrum(cosmology, ic, z=z)
    print("Done getting power spectrum")
    if filter is None:
        filter = powerspec._default_filter

    var = get_variance(smoothing_scales, filter, powerspec, snapshot)

    return var


class FieldFilter(object):
    def __init__(self):
        raise (RuntimeError, "Cannot instantiate directly, use a subclass instead")

    def M_to_R(self, M):
        """Return the mass scale (Msol h^-1) for a given length (Mpc h^-1 comoving)"""
        return (M / (self.gammaF * self.rho_bar)) ** 0.3333

    def R_to_M(self, R):
        """Return the length scale (Mpc h^-1 comoving) for a given spherical mass (Msol h^-1)"""
        return self.gammaF * self.rho_bar * R ** 3


class SharpKFilter(FieldFilter):

    def __init__(self, context):
        # FieldFilter.__init__(self)

        self.gammaF = 6 * np.pi ** 2
        self.rho_bar = pynbody.analysis.cosmology.rho_M(context, unit="Msol Mpc^-3 h^2 a^-3")

    def M_to_R(self, M):
        """Return the mass scale (Msol h^-1) for a given length (Mpc h^-1 comoving)"""
        return (M / (self.gammaF * self.rho_bar)) ** 0.3333

    @staticmethod
    def Wk(bla):
        return (bla <1)


def get_spherical_collapse_barrier(initial_parameters, z=99, delta_sc_0=1.686, output="delta", growth=None):
    if z != 0:
        if growth is None:
            D_a = pynbody.analysis.cosmology.linear_growth_factor(initial_parameters.initial_conditions, z=z)
        else:
            D_a = growth
        # D_a = 0.01
        delta_sc = delta_sc_0 * D_a

    elif z == 0:
        delta_sc = delta_sc_0

    else:
        raise AttributeError("Insert an integer for the redshift")

    if output == "rho/rho_bar":
        delta_sc += 1
    return delta_sc


def get_ellipsoidal_barrier_from_variance(variance, initial_parameters, z=99, beta=0.485, gamma=0.6, a=0.707,
                                          output="rho/rho_bar", delta_sc=None, delta_sc_0=1.686):
    if delta_sc is None:
        delta_sc = get_spherical_collapse_barrier(initial_parameters, z=z, output="delta", delta_sc_0=delta_sc_0)

    #var_squared = variance**2
    delta_sc_squared = delta_sc**2

    # a acts as a rescaler so one will recover B ~ delta_sc as the variance tends to zero only if a=1.
    B = np.sqrt(a) * delta_sc * (1 + (beta * ((variance / (a * delta_sc_squared))**gamma)))
    if output == "rho/rho_bar":
        return B + 1
    else:
        return B


def ellipsoidal_collapse_barrier(mass_smoothing_scales, ic, beta=0.485, gamma=0.615, a=0.707, z=99,
                                 cosmology="WMAP5", output="rho/rho_bar", delta_sc_0=1.686, filter=None):
    # beta = 0.47
    # a = 0.75 is favoured by Sheth & Tormen (2002) since in agreement
    # with Jenkins et al. (2001) halo mass function.

    variance = calculate_variance(mass_smoothing_scales, ic, z=z, cosmology=cosmology, filter=filter)
    B = get_ellipsoidal_barrier_from_variance(variance, ic, z=z, beta=beta, gamma=gamma, a=a, output=output,
                                              delta_sc_0=delta_sc_0)
    return B


def plot_barriers_vs_sigma(ellips_barrier, sc_barrier, variance, labels=None, ylabel="Collapse barrier"):
    sigma = np.sqrt(variance)
    colors = ["g", "b", "r"]

    if labels is None:
        labels = ["ST a=1", "ST a=0.75", "ST a=0.707"]

    for num in range(len(ellips_barrier)):
        label = labels[num]
        color = colors[num]
        plt.plot(sigma, ellips_barrier[num], label=label, color=color, ls="-")

    plt.plot(sigma, np.tile([sc_barrier], len(sigma)), label="spherical", color="k", ls="-")
    plt.xlabel(r"$\sigma (M)$")
    plt.ylabel(ylabel)
    plt.legend(loc="best")


if __name__ == "__main__":

    ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")
    w = window.WindowParameters(initial_parameters=ic, num_filtering_scales=50)

    ellips_barrier_m = ellipsoidal_collapse_barrier(w.smoothing_masses, ic, a=0.75, z=100, cosmology="WMAP5")
    ellips_barrier_r = ellipsoidal_collapse_barrier(w.smoothing_radii, ic, a=0.75, z=100, cosmology="WMAP5")
    # np.testing.assert_allclose(ellips_barrier_m, ellips_barrier_r, rtol=1e-06)

    ### PLOT BARRIERS VS SIGMA ###

    for z in [0, 100]:

        if z == 0:
            output = "contrast"
            ylabel = r"$\mathrm{Collapse} \mathrm{barrier} ( \delta )$"
            save_path = "/Users/lls/Desktop/barriers_vs_sigma_z=0.png"

        else:
            output = "rho/rho_bar"
            ylabel = r"$\mathrm{Collapse} \mathrm{barrier} ( \delta + 1)$"
            save_path = "/Users/lls/Desktop/barriers_vs_sigma_z=100.png"

        b_sc_ic = get_spherical_collapse_barrier(z=z)
        if z == 100:
            b_sc_ic = b_sc_ic + 1

        ell_b = []
        for a in [1, 0.75, 0.707]:
            b_a = ellipsoidal_collapse_barrier(w.smoothing_masses, ic, beta=0.485, gamma=0.615, a=a, z=z,
                                               cosmology="WMAP5", output=output)
            ell_b.append(b_a)

        var = calculate_variance(w.smoothing_masses, ic, z=z, cosmology="WMAP5")
        plot_barriers_vs_sigma(ell_b, b_sc_ic + 1, var, labels=None, ylabel=ylabel)
        plt.savefig(save_path)
