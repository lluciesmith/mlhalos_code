import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from sklearn.externals import joblib

saving_path = "/share/data1/lls/regression/shear/density_period_fix/"
path_features = "/share/data2/lls/features_w_periodicity_fix/"

# saving_path = "/share/data1/lls/regression/shear/"
# path_features = "/share/data1/lls/shear_quantities/quantities_id_ordered/"

testing_ids = np.load("/share/data1/lls/regression/in_halos_only/log_m_output/even_radii_and_random/testing_ids.npy")

den_features = np.lib.format.open_memmap(path_features + "ics_density_contrasts.npy", mode="r", shape=(256**3, 50))
# den_features = np.lib.format.open_memmap(path_features + "density_trajectories.npy", mode="r", shape=(256**3, 50))
den_testing = den_features[testing_ids]
del den_features

den_sub_ell = np.lib.format.open_memmap(path_features + "density_subtracted_ellipticity.npy", mode="r",
                                        shape=(256**3, 50))
ell_testing = den_sub_ell[testing_ids]
del den_sub_ell

den_sub_prol = np.lib.format.open_memmap(path_features + "density_subtracted_prolateness.npy", mode="r",
                                         shape=(256**3, 50))
prol_testing = den_sub_prol[testing_ids]
del den_sub_prol

halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
log_mass = np.log10(halo_mass[testing_ids])
np.save(saving_path + "true_halo_mass.npy", log_mass)
del halo_mass
del log_mass

clf = joblib.load(saving_path + "classifier/classifier.pkl")
X_test = np.column_stack((den_testing, ell_testing, prol_testing))

y_predicted = clf.predict(X_test)
np.save(saving_path + "predicted_halo_mass.npy", 10**y_predicted)
