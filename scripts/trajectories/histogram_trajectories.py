"""
Save sharp-k trajectories above k = 2pi/L.

Save the "cut" trajectories as in
`` /share/data1/lls/trajectories_sharp_k/ALL_traj_above_k_min_box.npy``

"""
import sys
sys.path.append("/home/lls/mlhalos_code")
import pynbody
import numpy as np
from scripts.hmf import hmf_tests as ht
from mlhalos import parameters
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.mlab as mlab
from scripts.hmf import likelihood


# Original smoothing scales

ic = parameters.InitialConditionsParameters()

m_bins = 10**np.arange(10, 15, 0.0033).view(pynbody.array.SimArray)
m_bins.units = "Msol h^-1"
r = ht.pynbody_m_to_r(m_bins, ic.initial_conditions)
k_smoothing = 1/r

boxsize = pynbody.array.SimArray([50])
k_min = 2*np.pi/boxsize

k_multiples = k_smoothing/k_min

den_con = np.lib.format.open_memmap("/share/data1/lls/trajectories_sharp_k/ALL_traj_1500_even_log_m.npy", mode="r",
                                    shape=(16777216, 1516))

values = [2, 5, 10]
for value in values:
    idx = (np.abs(k_multiples - value)).argmin()
    k_index = k_multiples[idx]
    den_index = den_con[:, idx]
    np.save("/share/data1/lls/trajectories_sharp_k/k_multiple_" + str(value) + "_density_contrasts.npy",
            np.concatenate(([k_index], den_index)))

# scp k_multiple_2_density_contrasts.npy lls@chewbacca.star.ucl.ac.uk:/Users/lls/Documents/CODE/stored_files
# /traj_examples/.
# scp k_multiple_5_density_contrasts.npy lls@chewbacca.star.ucl.ac.uk:/Users/lls/Documents/CODE/stored_files
# /traj_examples/.
# scp k_multiple_10_density_contrasts.npy lls@chewbacca.star.ucl.ac.uk:/Users/lls/Documents/CODE/stored_files
# /traj_examples/.



(mu_2, sig_2) = norm.fit(d_2[1:])
(mu_5, sig_5) = norm.fit(d_5[1:])
(mu_10, sig_10) = norm.fit(d_10[1:])

gauss_2 = mlab.normpdf(bins_2, mu_2, sig_2)
gauss_5 = mlab.normpdf(bins_5, mu_5, sig_5)
gauss_10 = mlab.normpdf(bins_10, mu_10, sig_10)

a_2, bins_2, p_2 = plt.hist(d_2[1:], histtype="step", label="n=2", bins=45, color="b", normed=True)
a_5, bins_5, p_5 = plt.hist(d_5[1:], histtype="step", label="n=5", bins=45, color="g", normed=True)
a_10, bins_10, p_10 = plt.hist(d_10[1:], histtype="step", label="n=10", bins=45, color="r", normed=True)

plt.plot(bins_2, gauss_2, 'b--', linewidth=2)
plt.plot(bins_5, gauss_5, 'g--', linewidth=2)
plt.plot(bins_10, gauss_10, 'r--', linewidth=2)

plt.legend(loc="best")
plt.xlabel(r"$\delta + 1$")
plt.yscale("log")


# Goodness of fit

def goodness_of_fit(data, expectation):
    chi_sq = np.sum(((data - expectation)**2)/expectation)
    return chi_sq

mid_bins_2 = (bins_2[1:] + bins_2[:-1])/2
mid_bins_5 = (bins_5[1:] + bins_5[:-1])/2
mid_bins_10 = (bins_10[1:] + bins_10[:-1])/2

g_2 = mlab.normpdf(mid_bins_2, mu_2, sig_2)
g_5 = mlab.normpdf(mid_bins_5, mu_5, sig_5)
g_10 = mlab.normpdf(mid_bins_10, mu_10, sig_10)

chi_2 = goodness_of_fit(a_2, g_2)
chi_5 = goodness_of_fit(a_5, g_5)
chi_10 = goodness_of_fit(a_10, g_10)


