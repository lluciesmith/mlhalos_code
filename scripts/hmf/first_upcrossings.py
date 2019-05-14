import matplotlib.pyplot as plt
import numpy as np
import pynbody

from mlhalos import parameters
from scripts.ellipsoidal import ellipsoidal_barrier as eb
from scripts.ellipsoidal import variance_analysis as va
from scripts.hmf import hmf_theory as ht
from scripts.hmf import predict_masses as pm


if __name__ == "__main__":
    initial_parameters = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")

    L = initial_parameters.boxsize_comoving
    k_min = 2*np.pi/L
    k_nyquist = k_min * np.sqrt(3) * initial_parameters.shape/2
    k_max = k_nyquist/2
    k = (np.logspace(np.log10(k_min), np.log10(k_max), num=50, endpoint=True)).view(pynbody.array.SimArray)
    k.units = "Mpc^-1 a^-1 h"

    r_th, v_th = va.top_hat_variance(k, initial_parameters, z=0)
    r_sk, v_sk = va.sharp_k_variance(k, initial_parameters, z=0)

    delta_sc = eb.get_spherical_collapse_barrier(initial_parameters, z=0, delta_sc_0=1.686, output="delta")

    nu_th = delta_sc/np.sqrt(v_th)
    nu_sk = delta_sc / np.sqrt(v_sk)

    f_PS_th = ht.get_nu_f_nu_theoretical(nu_th, "PS")
    f_PS_sk = ht.get_nu_f_nu_theoretical(nu_sk, "PS")

    f_ST_th = ht.get_nu_f_nu_theoretical(nu_th, "ST")
    f_ST_sk = ht.get_nu_f_nu_theoretical(nu_sk, "ST")

    pred_spherical = np.load("/Users/lls/Documents/CODE/stored_files/hmf/trajectories_sharp_k/correct_growth/"
                             "ALL_PS_predicted_masses_1500_even_log_m_spaced.npy")
    pred_spherical = pynbody.array.SimArray(pred_spherical)
    pred_spherical.units = "Msol h^-1"
    r_predicted = pm.pynbody_m_to_r(pred_spherical, initial_parameters.initial_conditions)
    r_check, variance_predicted = va.sharp_k_variance(r_predicted[~np.isnan(r_predicted)], initial_parameters)

    plt.plot(np.sqrt(nu_th), f_PS_th, color="b", label="TH")
    plt.plot(np.sqrt(nu_sk), f_PS_sk, color="b", label="SK", ls="--")
    plt.plot(np.sqrt(nu_th), f_ST_th, color="g", label="TH")
    plt.plot(np.sqrt(nu_sk), f_ST_sk, color="g", label="SK", ls="--")
    #plt.xscale("log")
    plt.legend(loc="best")

    v_th = eb.calculate_variance(r_predicted, initial_parameters, z=0)





