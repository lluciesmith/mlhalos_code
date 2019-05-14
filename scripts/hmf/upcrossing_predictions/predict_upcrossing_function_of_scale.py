import sys
sys.path.append("/home/lls/mlhalos_code")
from mlhalos import parameters
from mlhalos import density
import numpy as np
from scripts.hmf import predict_masses as mp
import pynbody
from scripts.ellipsoidal import ellipsoidal_barrier as eb


initial_parameters = parameters.InitialConditionsParameters()

L = initial_parameters.boxsize_comoving
k_min = 2 * np.pi / L
k_nyquist = k_min * np.sqrt(3) * initial_parameters.shape / 2
k_max = k_nyquist / 2
k = (np.logspace(np.log10(k_min), np.log10(k_max), num=800, endpoint=True)).view(pynbody.array.SimArray)
k.units = "Mpc^-1 a^-1 h"
k.sim = initial_parameters.initial_conditions

# smooth from smallest to largest smoothing radius scale
#
# k_smoothing = k.in_units("Mpc**-1")
# smoothing_radii = 1 / k_smoothing[::-1]
#
# d = density.Density(initial_parameters=initial_parameters, window_function="sharp k")
# smoothed_density = d.get_smooth_density_for_radii_list(initial_parameters, smoothing_radii)
#
# den_con = density.DensityContrasts.get_density_contrast(initial_parameters, smoothed_density)
# np.save("/share/data1/lls/upcrossings/small_box/d_contrasts.npy", den_con)
#
# # Need to invert order of k array in order to have index k[-1] < k[-2], etc..
#
# PS_k_predicted = mp.get_predicted_analytic_mass(k[::-1], initial_parameters, barrier="spherical", cosmology="WMAP5",
#                                                 trajectories=den_con)
# np.save("/share/data1/lls/upcrossings/small_box/PS_k_predicted.npy", PS_k_predicted)
#
#
# ST_k_predicted = mp.get_predicted_analytic_mass(k[::-1], initial_parameters, barrier="ST", cosmology="WMAP5",
#                                                 trajectories=den_con)
# np.save("/share/data1/lls/upcrossings/small_box/ST_k_predicted.npy", ST_k_predicted)



####### script to use if you already have density contrasts"

den_con = np.load("/share/data1/lls/upcrossings/small_box/d_contrasts.npy")

SK = eb.SharpKFilter(initial_parameters.initial_conditions)
ST_k_predicted = mp.get_predicted_analytic_mass(k[::-1], initial_parameters, barrier="ST", cosmology="WMAP5",
                                                trajectories=den_con, filter=SK)
np.save("/share/data1/lls/upcrossings/small_box/ST_k_predicted_barrier_sk_sigma.npy", ST_k_predicted)




