import sys
sys.path.append("/home/lls/mlhalos_code")
import pynbody
import numpy as np
from scripts.hmf import hmf_tests as ht
from mlhalos import parameters
import matplotlib.pyplot as plt


def k_scale_sample_variance_power_spectrum(boxsize, accuracy=0.1):
    minimum_k = np.sqrt(2*np.pi)/(accuracy * boxsize)
    return minimum_k


def sample_variance_fraction_Pk(k_smoothing, boxsize):
    v_pk = 2*np.pi/(boxsize**2 * k_smoothing**2)
    return np.sqrt(v_pk)


def find_mass_scale_from_k_scale(k_scale):
    r = 1 / k_scale
    r_comoving = r / 0.01
    m = ht.pynbody_r_to_m(r_comoving, ic.initial_conditions)
    m.units = "Msol h^-1"
    return m


if __name__ == "__main__":
    ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE")

    m_bins = 10 ** np.arange(10, 15, 0.0033).view(pynbody.array.SimArray)
    m_bins.units = "Msol h^-1"

    r = ht.pynbody_m_to_r(m_bins, ic.initial_conditions)
    assert r.units == 'Mpc a h**-1'
    r.sim = ic.initial_conditions
    r_physical = r.in_units("Mpc h**-1")

    k_smoothing = 1 / r_physical

    # small box

    boxsize = pynbody.array.SimArray([50])
    boxsize.units = "Mpc h**-1 a"
    boxsize.sim = ic.initial_conditions
    boxsize_physical = boxsize.in_units("Mpc h**-1")

    v_small = sample_variance_fraction_Pk(k_smoothing, boxsize_physical)

    k_min_10_small = k_scale_sample_variance_power_spectrum(boxsize_physical, accuracy=0.1)
    k_min_5_small = k_scale_sample_variance_power_spectrum(boxsize_physical, accuracy=0.05)
    m_5_small = find_mass_scale_from_k_scale(k_min_5_small)

    # large box

    b_large = pynbody.array.SimArray([200*0.01])
    b_large.units = "Mpc h**-1"

    v_large = sample_variance_fraction_Pk(k_smoothing, b_large)

    k_min_10_large = k_scale_sample_variance_power_spectrum(b_large, accuracy=0.1)
    k_min_5_large = k_scale_sample_variance_power_spectrum(b_large, accuracy=0.05)
    m_5_large = find_mass_scale_from_k_scale(k_min_5_large)


    def plot_radius_vs_sample_variance():

        plt.plot(r_physical, v_small, label="small", color="blue")
        plt.plot(r_physical, v_large, label="large", color="green")
        #plt.axvline(x=1/k_min_10_small, color="blue", ls="-")
        plt.axvline(x=1 / k_min_5_small, color="blue", ls="--")
        #plt.axvline(x=1/k_min_10_large, color="green", ls="-")
        plt.axvline(x=1 / k_min_5_large, color="green", ls="--")
        plt.plot([], ls="--", color="k", label=r"$5 \%$ error")
        # plt.plot([], ls="-", color="k", label=r"$10 \%$ error")
        plt.ylabel(r"$\sigma_P/P(k)$")
        plt.xlabel("r [Mpc/h]")
        plt.yscale("log")
        plt.xlim(r_physical.min(), 0.07957981855801369)
        plt.legend(loc="best")


    def plot_mass_vs_sample_variance():
        plt.plot(m_bins, v_small, label="small", color="blue")
        plt.plot(m_bins, v_large, label="large", color="green")
        #plt.axvline(x=1/k_min_10_small, color="blue", ls="-")
        plt.axvline(x=m_5_small, color="blue", ls="--")
        #plt.axvline(x=1/k_min_10_large, color="green", ls="-")
        plt.axvline(x=m_5_large, color="green", ls="--")
        plt.plot([], ls="--", color="k", label=r"$5 \%$ error")
        # plt.plot([], ls="-", color="k", label=r"$10 \%$ error")
        plt.ylabel(r"$\sigma_P/P(k)$")
        plt.xlabel(r"M [M$_{\odot}$ / h]")
        plt.yscale("log")
        plt.xscale("log")
        plt.legend(loc="best")
