import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/lls/Documents/mlhalos_code/")
from mlhalos import parameters
from scripts.hmf import halo_mass as hm


import glob
files = glob.glob("/Users/lls/Documents/CODE/stored_files/hmf/super_sampling/boxsize_50/std_07/PS_*.npy")
PS_ssc = np.array([np.load(file) for file in files])


ic_50 = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")
initial_snapshot = "/Users/lls/Documents/CODE/sim200/sim200.gadget3"
ic_200 = parameters.InitialConditionsParameters(initial_snapshot=initial_snapshot, path="/Users/lls/Documents/CODE/")

bins = np.arange(10, 15, 0.1)
mid_bins = (bins[1:] + bins[:-1])/2


# import glob
# files = glob.glob("/Users/lls/Documents/CODE/stored_files/hmf/super_sampling/boxsize_50/std_07/PS_*.npy")
# PS_ssc = np.array([np.load(file) for file in files])
PS_ssc_flat = PS_ssc.flatten()
PS_ssc_num = hm.get_empirical_number_halos(PS_ssc_flat, ic_50)
PS_ssc_small_vol = PS_ssc_num/(200**3)*(50**3)

# Original

PS_50 = np.load("/Users/lls/Documents/CODE/stored_files/hmf/trajectories_sharp_k/correct_growth/"
                "ALL_PS_predicted_masses_1500_even_log_m_spaced.npy")
PS_50_num = hm.get_empirical_number_halos(PS_50, ic_50)


# Super sampling L=50 Mpc h^-1 box

PS_ssc_previous = np.load("/Users/lls/Documents/CODE/stored_files/hmf/super_sampling/boxsize_50/PS_pred_mass_halo_3_test.npy")
PS_ssc_flat_previous = PS_ssc_previous.flatten()
num_ssc_previous = hm.get_empirical_number_halos(PS_ssc_flat_previous, ic_50)
PS_ssc_small_vol_previous = num_ssc_previous/(200**3)*(50**3)


# Super sampling L=200 Mpc h^-1 box

PS_200 = np.load("/Users/lls/Documents/CODE/stored_files/hmf/sim200/ALL_PS_predicted_masses.npy")
PS_200_num = hm.get_empirical_number_halos(PS_200, ic_200)
PS_200_num_small_sim = PS_200_num/(200**3)*(50**3)


### fractional differences ###
diff_200_50 = (PS_50_num - PS_200_num_small_sim)/PS_200_num_small_sim
diff_ssc_50 = (PS_ssc_small_vol - PS_200_num_small_sim)/PS_200_num_small_sim
diff_ssc_previous = (PS_ssc_small_vol_previous - PS_200_num_small_sim)/PS_200_num_small_sim


restr_m = 10**mid_bins[9:33]

plt.loglog(restr_m, PS_200_num_small_sim[9:33], label="large")
plt.scatter(restr_m, PS_ssc_small_vol[9:33], label="ssc", color="b")
plt.scatter(restr_m, PS_ssc_small_vol_previous[9:33], label="previous ssc", color="grey")
#plt.scatter(restr_m, PS_ssc_small_vol[9:33], color="k")
plt.loglog(restr_m, PS_50_num[9:33], label="orig")
plt.legend(loc="best")
plt.xlim(8*10**10, 2*10**13)

# plt.scatter(restr_m, diff_200_50[9:33], label="L=50 Mpc h^-1", color="b")
# plt.scatter(restr_m, diff_ssc_50[9:33], label="ssc L=50 Mpc h^-1", color="g")
# plt.scatter(restr_m, diff_ssc_previous[9:33], label="previous ssc", color="grey")
# plt.axhline(y=0, color="k", ls="--")
# plt.xscale("log")
# plt.legend(loc="best")
# plt.xlim(8*10**10, 2*10**13)
# # plt.ylim(-0.5, 0.5)
# plt.xlabel(r"$ \mathrm{M} [\mathrm{M}_{\odot} \mathrm{h}^{-1}]$")

plt.scatter(10**mid_bins, diff_200_50[9:33], label="L=50 Mpc h^-1", color="b")
plt.scatter(10**mid_bins, diff_ssc_50[9:33], label="ssc L=50 Mpc h^-1", color="g")
plt.scatter(10**mid_bins, diff_ssc_previous[9:33], label="previous ssc", color="grey")
plt.axhline(y=0, color="k", ls="--")
plt.xscale("log")
plt.legend(loc="best")
plt.xlim(8*10**10, 2*10**13)
# plt.ylim(-0.5, 0.5)
plt.xlabel(r"$ \mathrm{M} [\mathrm{M}_{\odot} \mathrm{h}^{-1}]$")
