import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import parameters
from scripts.hmf import predict_masses as mp
import pynbody

ic = parameters.InitialConditionsParameters()
ids_halo_5 = np.where(ic.final_snapshot['grp'] == 5)[0]

traj = np.load("/share/data1/lls/trajectories_sharp_k/ALL_traj_1500_even_log_m.npy")
id = np.random.choice(ids_halo_5, 1)
one_traj = traj[id[0]]
del traj

#traj_halo_5 = traj[ids_halo_5]
#del traj

std_50 = 0.26461567993066265
lin_growth = pynbody.analysis.cosmology.linear_growth_factor(ic.initial_conditions)
std_50_z_99 = std_50 * lin_growth

delta_l = np.random.normal(0, std_50_z_99, size=10000)
m_bins = 10**np.arange(10, 15, 0.0033).view(pynbody.array.SimArray)

#one_traj = np.random.choice(len(traj_halo_5), 1)
one_traj_prime = np.array([mp.get_predicted_analytic_mass(m_bins, ic, barrier="spherical",
                                                          trajectories=one_traj + delta_l[i]) for i in range(10000)])
np.save("/share/data1/lls/trajectories_sharp_k/super_sampling/boxsize_50/mass_distribution_id_"+ str(int(id)) +
        ".npy", one_traj_prime)

plt.plot(m_bins, traj_h_5[0], color="k", label="orig")
plt.plot(m_bins, traj_h_5[0] + delta_l[1], label=" + deltaL")
plt.plot(m_bins, traj_h_5[0] + delta_l[2], label=" - deltaL")
plt.axhline(y=(1.686*0.013) +1, color="k", ls="--")
plt.xlim(m_bins.min(), 10**14)
plt.xscale("log")
plt.ylim(0.95, 1.05)
plt.legend(loc="best")

plt.hist(np.log10(tr[~np.isnan(tr)]), bins=20, histtype="step")
plt.axvline(x=np.log10(pred_0), color="k", ls="-", lw=2)
plt.xlabel("log(mass)")


delta_l = np.random.normal(0, std_50_z_99, size=10000)

m_predicted_orig = np.load("/Users/lls/Documents/CODE/stored_files/hmf/trajectories_sharp_k/correct_growth"
                       "/ALL_predicted_masses_1500_even_log_m_spaced.npy")
ids_halo_5 = np.where(ic.final_snapshot['grp'] == 5)[0]
orig_halo_5 = m_predicted_orig[ids_halo_5]

tr = np.array([mp.get_predicted_analytic_mass(m_bins, ic, barrier="spherical", trajectories=traj_h_5[0] + delta_l[i])
               for i in range(10000)])
plt.hist(np.log10(orig_halo_5[~np.isnan(orig_halo_5)]), bins=20)
plt.hist(np.log10(tr[~np.isnan(tr)]), bins=20)
plt.axvline(x=np.log10(ic.halo[5]['mass'].sum()))