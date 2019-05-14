"""

Test the GBT classifier trained on ICs overdensity up to very high smoothing scales
on an independent box of L=50 Mpc/h a and N=256^3.

"""

import numpy as np
from mlhalos import parameters
from mlhalos import window
from mlhalos import density
import pynbody
from sklearn.externals import joblib

############### FEATURE EXTRACTION ###############

if __name__ == "__main__":
    path = "/home/lls"
    initial_params = parameters.InitialConditionsParameters(initial_snapshot="/home/app/reseed/IC.gadget3",
                                                            load_final=True,
                                                            final_snapshot="/home/app/reseed/snapshot_099")

    w = window.WindowParameters(initial_parameters=initial_params)
    d = density.DensityContrasts(initial_parameters=initial_params)

    m = np.linspace(np.log10(3e10), np.log10(1e15), 50)
    width = np.append(np.diff(m), np.diff(m)[-1])
    m_all_p = initial_params.initial_conditions["mass"].sum()
    m1 = np.arange(np.log10(1e15), np.log10(m_all_p * 5000), step=width[-1])[1:]

    m_all = 10 ** np.concatenate((m, m1))

    M = pynbody.array.SimArray(m_all)
    M.units = "Msol"
    r_smoothing = w.get_smoothing_radius_corresponding_to_filtering_mass(initial_params, M)

    den = d.get_smooth_density_for_radii_list(initial_params, r_smoothing)
    den_con = den / initial_params.mean_density


    halo_mass = np.load("/Users/lls/Documents/mlhalos_files/reseed50/regression/ic_traj_smoothing_scales_above_1e15"
                        "/halo_mass_particles.npy")
    in_ids = np.where(halo_mass > 0)[0]

    clf_2000 = joblib.load("/Users/lls/Desktop/clf.pkl")
    pred_50 = clf_2000.predict(den_con[in_ids])
    np.save("/Users/lls/Documents/mlhalos_files/reseed50/regression/ic_traj_smoothing_scales_above_1e15/predictions"
            ".npy", pred_50)
    np.save("/Users/lls/Documents/mlhalos_files/reseed50/regression/ic_traj_smoothing_scales_above_1e15/true_log_mass"
            ".npy", np.log10(halo_mass[in_ids]))
