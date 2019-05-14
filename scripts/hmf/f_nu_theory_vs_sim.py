import numpy as np
import pynbody

from scripts.hmf import hmf_simulation as hmf_sim
from scripts.hmf import hmf_empirical as hmf_emp
import scripts.hmf.hmf_theory
from mlhalos import parameters
from scripts.ellipsoidal import variance_analysis as va
from scripts.ellipsoidal import ellipsoidal_barrier as eb
import matplotlib.pyplot as plt
import scipy.interpolate


def get_variance_function(mass, initial_parameters, th=True):
    rho_M = pynbody.analysis.cosmology.rho_M(initial_parameters.initial_conditions, unit="Msol Mpc^-3 h^2 a^-3")
    r = (3 * mass / (4 * np.pi * rho_M)) ** (1 / 3)
    k = pynbody.array.SimArray(2 * np.pi / r)
    k.units = "Mpc^-1 a^-1 h"
    if th is True:
        r1, var = va.top_hat_variance(k, initial_parameters, z=0)
    else:
        r1, var = va.sharp_k_variance(k, initial_parameters, z=0)
    f = scipy.interpolate.interp1d(k, var)
    return f


def get_variance_as_function_mass(mass, initial_parameters):
    mass = mass.view(pynbody.array.SimArray)
    mass.units = "Msol h^-1"
    v = eb.calculate_variance(mass, initial_parameters, z=0)
    return v


#####################  Initial parameters + variance interpolation  ####################

# large sim

initial_snapshot = "/Users/lls/Documents/CODE/standard200/standard200.gadget3"
final_snapshot = "/Users/lls/Documents/CODE/standard200/snapshot_011"
ic_200 = parameters.InitialConditionsParameters(initial_snapshot=initial_snapshot,
                                                final_snapshot=final_snapshot, path="/Users/lls/Documents/CODE/")

# second_large_sim

initial_snapshot = "/Users/lls/Documents/CODE/new_sim200/standard200.gadget3"
final_snapshot = "/Users/lls/Documents/CODE/new_sim200/snapshot_011"
ic_200_2 = parameters.InitialConditionsParameters(initial_snapshot=initial_snapshot,
                                                final_snapshot=final_snapshot, path="/Users/lls/Documents/CODE/")

# small sim

ic_small = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")
rho_M = pynbody.analysis.cosmology.rho_M(ic_small.initial_conditions, unit="Msol Mpc^-3 h^2 a^-3")

# variance interpolation

mass_scales_variance = 10**np.arange(8, 16, 0.1)
var_function = get_variance_function(mass_scales_variance, ic_small)



#####################  simulation  ####################

m = 10**np.arange(10.5, 14, 0.1)
variance_small, f_small, err_small = hmf_sim.get_f_nu_simulation(ic_small, var_function, mass_bins=m, error=True)
v_small = (variance_small[1:] + variance_small[:-1])/2

m2 = 10**np.arange(11.5, 15.5, 0.1)
variance_large, f_large, err_large = hmf_sim.get_f_nu_simulation(ic_200, var_function, mass_bins=m2, error=True)
v_large = (variance_large[1:] + variance_large[:-1])/2

variance_large_2, f_large_2, err_large_2 = hmf_sim.get_f_nu_simulation(ic_200_2, var_function, mass_bins=m2, error=True)
v_large_2 = (variance_large[1:] + variance_large[:-1])/2


#################### theory ####################

delta_sc = eb.get_spherical_collapse_barrier(ic_small, z=0, delta_sc_0=1.686, output="delta")

v_th = np.arange(0.05, 10, step=0.01)
nu = delta_sc / np.sqrt(v_th)
f_PS = scripts.hmf.hmf_theory.get_nu_f_nu_theoretical(nu, "PS")
f_ST = scripts.hmf.hmf_theory.get_nu_f_nu_theoretical(nu, "ST")


# PLOT THEORY VS SIMULATION

var_tick = get_variance_as_function_mass(mass_scales_variance, ic_small)
x = np.log(1/np.sqrt(var_tick))
tick_function = scipy.interpolate.interp1d(x, mass_scales_variance)

def xticklab(var):
    return ["%.2f" % z for z in np.log10(tick_function(var))]


fig, ax1 = plt.subplots(figsize=(9,6))
ax2 = ax1.twiny()
ax1.plot(np.log(nu/delta_sc), np.log(f_PS), label="PS")
ax1.plot(np.log(nu/delta_sc), np.log(f_ST), label="ST")
ax1.errorbar(np.log(1/np.sqrt(v_small)), np.log(f_small), yerr=err_small/f_small,
             fmt="^", color="k", label="sim(small)", markersize=5)
ax1.errorbar(np.log(1/np.sqrt(v_large)), np.log(f_large), yerr=err_large/f_large,
             fmt="x", color="k", label="sim(large)", markersize=5)
ax1.errorbar(np.log(1/np.sqrt(v_large_2)), np.log(f_large_2), yerr=err_large_2/f_large_2,
             fmt="o", color="k", label="sim(large 2)", markersize=5)


ax2.set_xticks(ax1.get_xticks())
ax2.set_xticklabels(xticklab(ax1.get_xticks()), fontsize=15)
ax2.set_xlabel(r"$\log(M/M_{\odot})$")
#ax2.ticklabel_format(style='sci')

ax1.legend(loc="best", frameon=False)
ax1.set_xlabel(r"$\log [1/ \sigma ]$")
ax1.set_ylabel(r"$\log [f(\nu)]$")
ax1.set_ylim(-5, 0)
fig.subplots_adjust(top=0.88)
