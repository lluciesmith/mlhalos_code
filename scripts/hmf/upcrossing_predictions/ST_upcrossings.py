import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from scripts.hmf import predict_masses as ht
import pynbody
from mlhalos import parameters
from scripts.ellipsoidal import ellipsoidal_barrier as eb
from scripts.hmf import predict_masses as mp

#ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE", load_final=True)
ic = parameters.InitialConditionsParameters(load_final=True)

m_bins = 10**np.arange(10, 15, 0.0033).view(pynbody.array.SimArray)
m_bins.units = "Msol h^-1"

r_smoothing = ht.pynbody_m_to_r(m_bins, ic.initial_conditions)
den_con = np.load("/share/data1/lls/trajectories_sharp_k/ALL_traj_1500_even_log_m.npy")

# ST barrier with \sigma(R) calculated with top-hat window function

b_z = eb.ellipsoidal_collapse_barrier(r_smoothing, ic, beta=0.485, gamma=0.615, a=0.707, z=99,
                                 cosmology="WMAP5", output="rho/rho_bar", delta_sc_0=1.686, filter=None)
pred_radii = mp.record_mass_first_upcrossing(b_z, r_smoothing, trajectories=den_con)
np.save("/share/data1/lls/upcrossings/ST_r_upcrossing_TH_sigma.npy",pred_radii)


# ST barrier with \sigma(R) calculated with sharp-k window function

SK = eb.SharpKFilter(ic.initial_conditions)
b_z_SK = eb.ellipsoidal_collapse_barrier(r_smoothing, ic, beta=0.485, gamma=0.615, a=0.707, z=99,
                                 cosmology="WMAP5", output="rho/rho_bar", delta_sc_0=1.686, filter=SK)
pred_radii_SK = mp.record_mass_first_upcrossing(b_z_SK, r_smoothing, trajectories=den_con)
np.save("/share/data1/lls/upcrossings/ST_r_upcrossing_SK_sigma.npy",pred_radii_SK)

# SK filter barrier is lower than the top hat one!
# plt.plot(r_smoothing, b_z, label="TH")
# plt.plot(r_smoothing, b_z_SK, label="SK")
# plt.legend()
# plt.xlabel(r"$r$[Mpc h$^{-1}$ a]")
# plt.ylabel("$b_{\mathrm{ST}}$")
