"""
Test what happens when you increase the number of training examples.
In particular, in increasing the number of particles at the edges of the distribution.

"""
import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from mlhalos import machinelearning as ml


# halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
# log_halo_mass = np.log10(halo_mass[halo_mass > 0])
#
# training_ids = np.load("/share/data1/lls/regression/in_halos_only/log_m_output/even_radii_and_random/training_ids.npy")
#
# ids_in_halo = np.arange(256**3)[halo_mass > 0]
#
# remaining_ids = ids_in_halo[~np.in1d(ids_in_halo, training_ids)]
# additional_training = []
#
# bins_plotting = np.linspace(log_halo_mass.min(), log_halo_mass.max(), 15, endpoint=True)
# for i in range(len(bins_plotting) -1):
#
#     ids_i = remaining_ids[(np.log10(halo_mass[remaining_ids]) >= bins_plotting[i]) &
#                           (np.log10(halo_mass[remaining_ids]) <= bins_plotting[i+1])]
#
#     if i < 3:
#         additional_training.append(np.random.choice(ids_i, 20000, replace=False))
#
#     elif i >=3 and i <=11:
#         additional_training.append(np.random.choice(ids_i, 10000, replace=False))
#
#     else:
#         additional_training.append(np.random.choice(ids_i, 15000, replace=False))
#
#
# augmented_training = np.concatenate((training_ids, np.concatenate(additional_training)))
# halo_mass_training = np.log10(halo_mass[augmented_training])
# np.save("/share/data1/lls/regression/in_halos_only/log_m_output/larger_training_set/ids_training_set_large.npy",
#         augmented_training)
# np.save("/share/data1/lls/regression/in_halos_only/log_m_output/larger_training_set/halos_training_set_large.npy",
#         halo_mass_training)
#
# testing_ids = ids_in_halo[~np.in1d(ids_in_halo, augmented_training)]
# halo_mass_testing = np.log10(halo_mass[testing_ids])
# np.save("/share/data1/lls/regression/in_halos_only/log_m_output/larger_training_set/ids_testing_set_large.npy",
#         testing_ids)
# np.save("/share/data1/lls/regression/in_halos_only/log_m_output/larger_training_set/halos_testing_set_large.npy",
#         halo_mass_testing)

augmented_training = np.load("/share/data1/lls/regression/in_halos_only/log_m_output/larger_training_set/ids_training_set_large.npy")
halo_mass_training = np.load("/share/data1/lls/regression/in_halos_only/log_m_output/larger_training_set/halos_training_set_large.npy")

testing_ids = np.load("/share/data1/lls/regression/in_halos_only/log_m_output/larger_training_set/ids_testing_set_large.npy")
halo_mass_testing = np.load("/share/data1/lls/regression/in_halos_only/log_m_output/larger_training_set/halos_testing_set_large.npy")


# create mock features with different noise levels

dup = np.copy(halo_mass_training)
dup1 = np.tile(dup, (50, 1)).transpose()

noise_04 = np.random.normal(0, 2.7, size=[len(halo_mass_training), 50])
feat_04_corr = dup1 + noise_04

# noise_07 = np.random.normal(0, 1.2, [len(halo_mass_training), 50])
# feat_07_corr = dup1 + noise_07

# training

print("Start training...")

rf = RandomForestRegressor()
param_grid = {'max_depth': [100, 200, None],  'max_features': [5, 10], 'n_estimators': [600, 1000]}
rf_CV = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1, scoring="neg_mean_squared_error")

#rf_CV.fit(np.column_stack((feat_07_corr, feat_04_corr)), halo_mass_training.reshape(-1, 1))
rf_CV.fit(feat_04_corr, halo_mass_training.reshape(-1, 1))

ml.write_to_file_cv_results("/share/data1/lls/regression/in_halos_only/log_m_output/larger_training_set"
                            "/04_only/cv_results.txt",
                         rf_CV)
joblib.dump(rf_CV, "/share/data1/lls/regression/in_halos_only/log_m_output/larger_training_set/04_only/clf.pkl")


# predictions

dup = np.copy(halo_mass_testing)
dup1 = np.tile(dup, (50, 1)).transpose()

noise_04 = np.random.normal(0, 2.7, size=[len(halo_mass_testing), 50])
test_feat_04_corr = dup1 + noise_04

# noise_07 = np.random.normal(0, 1.2, [len(halo_mass_testing), 50])
# test_feat_07_corr = dup1 + noise_07

pred = rf_CV.predict(test_feat_04_corr)
np.save("/share/data1/lls/regression/in_halos_only/log_m_output/larger_training_set/04_only/predicted_halos_test.npy"
        "", pred)