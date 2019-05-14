import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
import pynbody
from multiprocessing import Pool
from mlhalos import parameters
from scripts.hmf.larger_sim import subbox as sb
from scripts.hmf import predict_masses as mp


initial_snapshot = "/home/app/scratch/sim200.gadget3"
# final_snapshot = "/home/app/scratch/snapshot_011"
ic_200 = parameters.InitialConditionsParameters(initial_snapshot=initial_snapshot)

rhoM = pynbody.analysis.cosmology.rho_M(ic_200.initial_conditions, unit="Msol kpc**-3")
# rhoM = ic_200.mean_density
sim = ic_200.initial_conditions

rho_200 = np.load("/home/lls/stored_files/mean_densities/rho_200.npy")
sim['rho'] = rho_200

particle = 62155961
length_subbox = 50 * 0.01 / 0.701 * 10 ** 3

# Method1

# ids_subbox = sb.get_ids_subbox_centered_on_particle(sim, particle, length_subbox)
# den_subbox = np.mean(sim[ids_subbox]['rho'])
# delta_subbox = (den_subbox - rhoM) / rhoM

# Method2


std_50 = np.sqrt(0.0700214580652)
lin_growth = pynbody.analysis.cosmology.linear_growth_factor(ic_200.initial_conditions)
std_50_z_99 = std_50 * lin_growth
delta_mean = np.random.normal(0, std_50_z_99, size=100)

m_bins = 10 ** np.arange(10, 16, 0.0033).view(pynbody.array.SimArray)
m_bins.units = "Msol h^-1"
m = m_bins[::2]

for j in range(len(delta_mean)):
    delta_subbox = delta_mean[j]

    def predict_ssc(i):
        t = np.load("/share/data1/lls/sim200/subboxes/subbox_62155961/trajectories/trajectories_ids_" + str(i) + ".npy")
        t_ssc = t + delta_subbox
        del t
        pred_mass_PS = mp.get_predicted_analytic_mass(m, ic_200, barrier="spherical", cosmology="WMAP5",
                                                        trajectories=t_ssc)
        pred_mass_ST = mp.get_predicted_analytic_mass(m, ic_200, barrier="ST", cosmology="WMAP5",
                                                        trajectories=t_ssc)
        return pred_mass_PS, pred_mass_ST

    pool = Pool(processes=60)
    f = predict_ssc

    numbers = np.arange(193)
    PS=[]
    ST=[]
    for PS_i, ST_i in pool.map(f, numbers):
        PS.append(PS_i)
        ST.append(ST_i)

    # pool.join()
    pool.close()

    PS = np.array(PS)
    ST = np.array(ST)

    pred_sc_flat_PS = np.array([item for simarray in PS for item in simarray])
    pred_sc_flat_ST = np.array([item for simarray in ST for item in simarray])

    np.save("/share/data1/lls/sim200/subboxes/subbox_62155961/pred_ssc_PS_delta_ " + str(delta_subbox) + ".npy",
            pred_sc_flat_PS)
    np.save("/share/data1/lls/sim200/subboxes/subbox_62155961/pred_ssc_ST " + str(delta_subbox) + ".npy",
            pred_sc_flat_ST)
    del pred_sc_flat_PS
    del pred_sc_flat_ST

