import sys
sys.path.append('/home/lls/mlhalos_code')
import numpy as np
from mlhalos import parameters
from scripts.hmf import predict_masses as ht
from mlhalos import density
import pynbody
from scripts.ellipsoidal import ellipsoidal_barrier as eb
from scripts.hmf import predict_masses as mp


ic = parameters.InitialConditionsParameters(initial_snapshot="/home/app/reseed/IC.gadget3",
                                            final_snapshot="/home/app/reseed/snapshot_099",
                                            load_final=True, min_halo_number=0, max_halo_number=400,
                                            min_mass_scale=3e10, max_mass_scale=1e15)

ic.max_halo_number = 409

m_bins = 10**np.arange(10, 15, 0.0033).view(pynbody.array.SimArray)
m_bins.units = "Msol h^-1"

r_comoving = ht.pynbody_m_to_r(m_bins, ic.initial_conditions)
assert r_comoving.units == 'Mpc a h**-1'
r_physical = r_comoving * ic.initial_conditions.properties['a'] / ic.initial_conditions.properties['h']
r_physical.units = "Mpc"

r_smoothing = np.array(r_physical)

################## sharp-k trajectories ###########################

den = density.Density(initial_parameters=ic, window_function="sharp k")
smoothed_density = den.get_smooth_density_for_radii_list(ic, r_smoothing)

den_con = density.DensityContrasts.get_density_contrast(ic, smoothed_density)
np.save("/share/data1/lls/reseed50/sharp_k_filter/all_trajectories_sharp_k.npy", den_con)

del smoothed_density
del den
del r_physical

##################  PRESS - SCHECHTER  ##################

den_con = np.load("/share/data1/lls/reseed50/sharp_k_filter/all_trajectories_sharp_k.npy")

pred_radii_PS = mp.get_predicted_analytic_mass(r_comoving, ic, barrier="spherical", trajectories=den_con)
np.save("/share/data1/lls/reseed50/sharp_k_filter/PS_r_upcrossing.npy",pred_radii_PS)

##################  SHETH - TORMEN  ##################

b_z = eb.ellipsoidal_collapse_barrier(r_comoving, ic, beta=0.485, gamma=0.615, a=0.707, z=99,
                                 cosmology="WMAP5", output="rho/rho_bar", delta_sc_0=1.686, filter=None)
pred_radii_TH = mp.record_mass_first_upcrossing(b_z, r_comoving, trajectories=den_con)
np.save("/share/data1/lls/reseed50/sharp_k_filter/ST_r_upcrossing_TH_sigma.npy",pred_radii_TH)


# ST barrier with \sigma(R) calculated with sharp-k window function

SK = eb.SharpKFilter(ic.initial_conditions)
b_z_SK = eb.ellipsoidal_collapse_barrier(r_comoving, ic, beta=0.485, gamma=0.615, a=0.707, z=99,
                                 cosmology="WMAP5", output="rho/rho_bar", delta_sc_0=1.686, filter=SK)
pred_radii_SK = mp.record_mass_first_upcrossing(b_z_SK, r_comoving, trajectories=den_con)
np.save("/share/data1/lls/f_sigma_ST_large.pdf/reseed50/sharp_k_filter/ST_r_upcrossing_SK_sigma.npy",pred_radii_SK)