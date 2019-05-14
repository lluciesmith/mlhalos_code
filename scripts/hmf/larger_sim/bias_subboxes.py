import sys
sys.path.append("/Users/lls/Documents/mlhalos_code")
#sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import parameters
import pynbody
import matplotlib.pyplot as plt
from scripts.hmf.larger_sim import subbox as sb
from scripts.hmf import halo_mass as hm
from multiprocessing import Pool


# HYPATIA - CALCULATE THE HMF RESPONSE FROM SUBBOXES OF L=200 MPC BOX

initial_snapshot = "/home/app/scratch/sim200.gadget3"
# final_snapshot = "/home/app/scratch/snapshot_011"
ic_200 = parameters.InitialConditionsParameters(initial_snapshot=initial_snapshot)

pred = np.load("/share/data1/lls/sim200/ALL_PS_predicted_masses.npy")


# Box quantities

rhoM = pynbody.analysis.cosmology.rho_M(ic_200.initial_conditions, unit="Msol kpc**-3")
# rhoM = ic_200.mean_density
sim = ic_200.initial_conditions

pred_no_nan = pred[~np.isnan(pred)]
num_tot = hm.get_empirical_number_halos(pred_no_nan, ic_200)
num_den_tot = num_tot/(200**3)

# subbox


def get_hmf_response_in_subbox(particle):
    ids_subbox = sb.get_ids_subbox_centered_on_particle(sim, particle, length_subbox)
    sb_pred = pred[ids_subbox]

    den_subbox = np.mean(sim[ids_subbox]['rho'])
    delta_subbox = (den_subbox - rhoM) / rhoM

    sb_pred_no_nan = sb_pred[~np.isnan(sb_pred)]
    num_subbox = hm.get_empirical_number_halos(sb_pred_no_nan, ic_200)
    # Correct for volume subbox ?
    volume_original = 50**3
    volume_modified = volume_original * (1 - delta_subbox)
    num_den_subbox_modified= num_subbox / volume_modified
    num_den_subbox_original = num_subbox / volume_original

    n_diff_orig = (num_den_subbox_original - num_den_tot) / num_den_tot
    n_diff_modified = (num_den_subbox_modified - num_den_tot) / num_den_tot
    return delta_subbox, n_diff_orig, n_diff_modified


particle_ids = np.random.choice(np.arange(512**3), 10)
length_subbox = 50 * 0.01 / 0.701 * 10 ** 3

pool = Pool(processes=4)
f = get_hmf_response_in_subbox

deltas=[]
diffsO=[]
diffsM=[]
for delta, diff_orig, diff_modified in pool.map(f, particle_ids):
    deltas.append(delta)
    diffsO.append(diff_orig)
    diffsM.append(diff_modified)
pool.join()
pool.close()

a = np.column_stack((np.array(deltas), np.array(diffs)))
np.save("/share/data1/lls/sim200/subboxes/modify_volume/delta_hmf_diffs.npy", a)
# np.save("/share/data1/lls/sim200/subboxes/delta_hmf_diffs.npy", a)


# MAC PRO - CHECK THAT L=50 to L=200 HMF VARIATION IS CONSISTENT WITH FLUCTUATIONS ABOVE

initial_snapshot = "/Users/lls/Documents/CODE/standard200/standard200.gadget3"
final_snapshot = "/Users/lls/Documents/CODE/standard200/snapshot_011"
ic_200 = parameters.InitialConditionsParameters(initial_snapshot=initial_snapshot,
                                                final_snapshot=final_snapshot, path="/Users/lls/Documents/CODE/")
ic_50 = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")

pred_200 = np.load("/Users/lls/Documents/CODE/stored_files/hmf/sim200/ALL_PS_predicted_masses.npy")
pred_50 = np.load("/Users/lls/Documents/CODE/stored_files/hmf/trajectories_sharp_k/correct_growth/"
                  "/ALL_PS_predicted_masses_1500_even_log_m_spaced.npy")
pred_200_no_nan = pred_200[~np.isnan(pred_200)]
pred_50_no_nan = pred_50[~np.isnan(pred_50)]

hmf_200 = hm.get_empirical_number_halos(pred_200_no_nan, ic_200)/(200**3)
hmf_50 = hm.get_empirical_number_halos(pred_50_no_nan, ic_50)/(50**3)
diff_50_200 = (hmf_50 - hmf_200)/hmf_200


delta_diff = np.load("/Users/lls/Documents/CODE/stored_files/hmf/sim200/delta_hmf_diffs.npy")
d = delta_diff.transpose()

for i in range(45):
    plt.hist(d[1+i, :], bins=20, histtype="step", normed=True)
    plt.scatter(diff_50_200[i], 0, marker="x")
    plt.xlabel("Mass bin " + str(i))
    plt.savefig("/Users/lls/Desktop/hmf_response/mass_bin_" + str(i) + ".png")
    plt.clf()


# for particle_id in particle_ids:
#     particle_id = particle_ids[i]
#     ids_subbox = sb.get_ids_subbox_centered_on_particle(sim, particle_id, length_subbox)
#     sb_pred = pred[ids_subbox]
#
#     den_subbox = np.mean(sim[ids_subbox]['rho'])
#     delta_subbox = (den_subbox - rhoM)/rhoM
#
#     num_subbox = hm.get_empirical_number_halos(sb_pred, ic_200)
#     num_den_subbox = num_subbox/(50**3)
#
#     n_diff = (num_den_subbox - num_den_tot)/num_den_tot
#
#     delta_L.append(delta_subbox)
#     diff_n.append(n_diff)

# plot

# bins = np.arange(10, 15, 0.1)
# mid_bins = (bins[1:] + bins[:-1])/2
#
# plt.loglog(10**mid_bins, num_subbox/(50**3), label="sub-box")
# plt.loglog(10**mid_bins, num_tot/(200**3), label="total")
# plt.legend(loc="best")
#
# plt.plot(10**mid_bins, (num_den_subbox - num_den_tot)/num_den_tot)
# plt.xscale("log")