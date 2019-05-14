import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/lls/Documents/mlhalos_code")
import numpy as np
from mlhalos import parameters
import pynbody
from scripts.hmf import predict_masses as mp

ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")
num_realisations = 10000

std_50 = 0.26461567993066265
lin_growth = pynbody.analysis.cosmology.linear_growth_factor(ic.initial_conditions)
std_50_z_99 = std_50 * lin_growth
delta_l = np.random.normal(0, std_50_z_99, size=num_realisations)

traj_h_5 = np.load("/Users/lls/Documents/CODE/stored_files/hmf/trajectories_sharp_k/traj_halo_5.npy")

m_predicted_orig = np.load("/Users/lls/Documents/CODE/stored_files/hmf/trajectories_sharp_k/correct_growth"
                       "/ALL_PS_predicted_masses_1500_even_log_m_spaced.npy")
orig_halo_5 = m_predicted_orig[np.where(ic.final_snapshot['grp'] == 5)[0]]

m_bins = 10**np.arange(10, 15, 0.0033).view(pynbody.array.SimArray)

# # Trajectories examples
#
# plt.plot(m_bins, traj_h_5[0], color="k", label="orig")
# plt.plot(m_bins, traj_h_5[0] + delta_l[1], label=" + deltaL")
# plt.plot(m_bins, traj_h_5[0] + delta_l[2], label=" - deltaL")
# plt.axhline(y=(1.686*0.013) +1, color="k", ls="--")
# plt.xlim(m_bins.min(), 10**14)
# plt.xscale("log")
# plt.ylim(0.95, 1.05)
# plt.legend(loc="best")
#
# # Mass distribution first ID of halo 5
#
# tr = np.array([mp.get_predicted_analytic_mass(m_bins, ic, barrier="spherical", trajectories=traj_h_5[0] + delta_l[i])
#                for i in range(num_realisations)])
# plt.hist(np.log10(tr[~np.isnan(tr)]), bins=20, histtype="step")
# plt.axvline(x=np.log10(orig_halo_5[0]), color="k", ls="-", lw=2)
# plt.xlabel("log(mass)")
#
# # Mass distribution halo 5
#
#
# m_distribution = np.array([mp.get_predicted_analytic_mass(m_bins, ic, barrier="spherical", trajectories=traj_h_5+
#                                                                                                     delta_l[i])
#                            for i in range(num_realisations)])
#
#
# tr_mean_2000 = np.array([np.mean(tr[:, i][~np.isnan(tr[:,i])]) for i in range(len(tr[0]))])
#
# tr_f = np.mean(tr, axis=0)
#
# n, b, p = plt.hist(np.log10(orig_halo_5[~np.isnan(orig_halo_5)]), bins=20, label="orig", histtype="step")
# plt.hist(np.log10(tr_mean_2000[~np.isnan(tr_mean_2000)]), bins=b, histtype="step")
# plt.axvline(x=np.log10(ic.halo[5]['mass'].sum()), color="k", label="true")
# plt.xlabel("log(mass)")
# plt.legend(loc="best")

def plot_traj(i, delt=0, color="k"):
    if isinstance(i, (np.ndarray, list)):
        tr_new = (traj_h_5*(1 + delt)) + delt
        [plt.plot(m_bins[traj_h_5[j] != 0], tr_new[j][traj_h_5[j] != 0], color=color) for j in i]
    else:
        tr_new = (traj_h_5[i]*(1+delt)) + delt
        plt.plot(m_bins[traj_h_5 != 0], tr_new[traj_h_5!= 0], color=color)
    plt.axhline(y=(1.686 * 0.013) + 1, color="k", ls="--")
    plt.axvline(x=ic.halo[5]['mass'].sum()* 0.701, color="k", ls="--")
    plt.xscale("log")
    #plt.show()
