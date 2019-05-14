import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import machinelearning as ml
from mlhalos import parameters
import pynbody

# Get training set
saving_path = "/share/data1/lls/reseed50/"
path_simulation = "/home/app/reseed/"

initial_params = parameters.InitialConditionsParameters(initial_snapshot=path_simulation + "IC.gadget3",
                                                        final_snapshot=path_simulation + "snapshot_099",
                                                        load_final=True, min_halo_number=0, max_halo_number=400,
                                                        min_mass_scale=3e10, max_mass_scale=1e15)
# Get halo mass for each particle

halo_id_particles = initial_params.final_snapshot['grp']
halo_num = np.unique(halo_id_particles)[1:]

halo_mass_particles = np.zeros(len(halo_id_particles))
for i in halo_num:
    halo_mass_particles[np.where(halo_id_particles == i)[0]] = initial_params.halo[i]["mass"].sum()

halo_mass_particles = pynbody.array.SimArray(halo_mass_particles)
halo_mass_particles.units = initial_params.halo[0]["mass"].units
np.save(saving_path + "halo_mass_particles.npy", halo_mass_particles)

# Get training set

in_halos_ids = np.where(halo_mass_particles > 0)[0]
training_ids = np.random.choice(in_halos_ids, 100000, replace=False)
np.save(saving_path + "regression/100k_training_set.npy", training_ids)

eig_0 = np.lib.format.open_memmap(saving_path + "features/inertia/eigenvalues_0.npy", mode="r", shape=(256**3, 50))
eig_0_training = eig_0[training_ids]
del eig_0

log_mass = np.log10(halo_mass_particles[training_ids])
del halo_mass_particles


# train

training_features = np.column_stack((eig_0_training, log_mass))
print(training_features.shape)

cv = True
third_features = int((training_features.shape[1] -1)/3)
param_grid = {"n_estimators": [800, 1000, 1300],
              "max_features": [third_features, "sqrt", 5, 10],
              "min_samples_leaf": [5, 15],
              #"criterion": ["mse", "mae"],
              }

clf = ml.MLAlgorithm(training_features, method="regression", cross_validation=cv, split_data_method=None, n_jobs=60,
                     save=True, path=saving_path + "regression/classifier/classifier.pkl", param_grid=param_grid)
if cv is True:
    print(clf.best_estimator)
    print(clf.algorithm.best_params_)
    print(clf.algorithm.best_score_)

np.save(saving_path + "regression/f_imp.npy", clf.feature_importances)

# # test
#
# y_predicted = clf.algorithm.predict(eig_0_testing)
# np.save(saving_path + "predicted_halo_mass.npy", 10**y_predicted)
