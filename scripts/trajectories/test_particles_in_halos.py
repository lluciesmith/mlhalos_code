import numpy as np
import sys
mlhalos_path = "/Users/lls/Documents/mlhalos_code"
sys.path.append(mlhalos_path)
from mlhalos import parameters
from scripts.hmf import mass_predictions_ST as mp
from scripts.hmf import halo_mass as hm
import matplotlib.pyplot as plt
from scripts.hmf import hmf_tests


smoothing_radii_50 = np.linspace(0.0057291, 0.20, 50)

t = np.load("/Users/lls/Documents/CODE/stored_files/hmf/trajectories_sharp_k"
            "/smoothed_density_contrast_50_particles_in_halos.npy")
# t1 = np.load("/share/data1/lls/trajectories_sharp_k/particles_in_halos/"
#              "smoothed_density_contrast_250_particles_in_halos.npy")

ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")

pred_sph = mp.from_radius_predict_mass(smoothing_radii_50, ic, barrier="spherical", traject=t)
r_comoving = hmf_tests.radius_comoving_h_units(smoothing_radii_50, ic.initial_conditions)
smoothing_m = hmf_tests.pynbody_r_to_m(r_comoving, ic.initial_conditions)

mass_sk = ic.mean_density * 6 * np.pi**2 * smoothing_radii_50**3
pred_sk = mp.get_predicted_analytic_mass(mass_sk, ic, barrier="spherical", cosmology="WMAP5",
                                            trajectories=t)

pred_sk_h = pred_sk * ic.initial_conditions.properties['h']
pred_sk_h.units = "Msol h^-1"

hmf_sk = hm.get_empirical_number_halos(pred_sk_h, ic, log_m_bins=20)
hmf_th = hm.get_empirical_number_halos(pred_sph, ic, log_m_bins=20)

m_th, num_th = hm.theoretical_number_halos(ic, kernel="PS")


bins = np.arange(10, 15, 0.1)
mid_bins = (bins[1:] + bins[:-1])/2

plt.loglog(10**mid_bins, num_th, color="k", label="theory")
plt.loglog(10**mid_bins, hmf_th, label="top hat")
plt.loglog(10**mid_bins, hmf_sk, label="sharp k")
plt.legend(loc="best")
plt.savefig('/Users/lls/Dropbox/top_hat_vs_sk_50_particles_in_halos.pdf')
