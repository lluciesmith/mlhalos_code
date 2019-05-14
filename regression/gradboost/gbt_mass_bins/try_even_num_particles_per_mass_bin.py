import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor


def get_loss_vs_iterations(clf, features, truth, percent=False):
    mae = np.zeros(len(clf.estimators_), )
    for i, y_pred in enumerate(clf.staged_predict(features)):
        if percent is True:
            mae[i] = np.mean(np.abs((truth - y_pred) / truth)) * 100
        else:
            mae[i] = mean_absolute_error(truth, y_pred)
    return mae


saving_path = "/share/data2/lls/regression/gradboost/halos_range_115_125/even_num_per_mass_bin/"
halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
traj = np.load("/share/data2/lls/features_w_periodicity_fix/ics_density_contrasts.npy")

training_ids = np.load("/share/data2/lls/regression/gradboost/randomly_sampled_training/ic_traj/nest_2000_lr006/"
                       "training_ids.npy")
testing_ids = np.load("/share/data2/lls/regression/gradboost/randomly_sampled_training/ic_traj/nest_2000_lr006/"
                       "testing_ids.npy")

ids_between_115_125 = np.where((np.log10(halo_mass[training_ids]) > 11.5) &
                                        (np.log10(halo_mass[training_ids]) <= 12.5))[0]

### ignore last feature due to sampling of window function in fourier space

training_even = []
n, bins = np.histogram(np.log10(halo_mass[training_ids[ids_between_115_125]]), bins=11)
num_particles = n.min()
for i in range(len(bins) -1):
    n_bin = np.where((np.log10(halo_mass[training_ids[ids_between_115_125]]) >= bins[i])
                     & (np.log10(halo_mass[training_ids[ids_between_115_125]]) < bins[i+1]))[0]
    training_even.append(np.random.choice(ids_between_115_125[n_bin], num_particles, replace=False))

training_even = np.array(training_even).flatten()

features_training = traj[training_ids[training_even], :-1]
truth_training = np.log10(halo_mass[training_ids[training_even]])

ids_testing_between_115_125 = np.where((np.log10(halo_mass[testing_ids]) > 11.5) &
                                        (np.log10(halo_mass[testing_ids]) <= 12.5))[0]


features_testing = traj[testing_ids[ids_testing_between_115_125], :-1]
truth_testing = np.log10(halo_mass[testing_ids[ids_testing_between_115_125]])

### take same number of particles per halo

all_ids = np.where((np.log10(halo_mass) > 12.5) & (np.log10(halo_mass) <= 13.5))[0]

log_h = np.log10(halo_mass[all_ids])
halos_unique = np.unique(log_h)
training_unique = []
for halo in halos_unique:
    n = np.where(log_h == halo)[0]
    training_unique.append(np.random.choice(n, 400, replace=False))

training_unique = np.random.permutation(np.array(training_unique).flatten())

features_training = traj[all_ids[training_unique], :-1]
truth_training = log_h[training_unique]




# train one classifier with depth 10

clf_10 = GradientBoostingRegressor(n_estimators=10, max_depth=10, max_features=0.8, warm_start=True, subsample=0.8,
                                  learning_rate=0.01)
clf_10.fit(features_training, truth_training)
np.save(saving_path + "imp_10_trees_depth_10.npy", clf_10.feature_importances_)

clf_10.n_estimators += 90
clf_10.fit(features_training, truth_training)
np.save(saving_path + "imp_90_trees_depth_10.npy", clf_10.feature_importances_)

clf_10.n_estimators += 50
clf_10.fit(features_training, truth_training)
np.save(saving_path + "imp_150_trees_depth_10.npy", clf_10.feature_importances_)

clf_10.n_estimators += 250
clf_10.fit(features_training, truth_training)
np.save(saving_path + "imp_400_trees_depth_10.npy", clf_10.feature_importances_)
joblib.dump(clf_10, saving_path + "clf_10_depth.pkl")

mae_train_10 = get_loss_vs_iterations(clf_10, features_training, truth_training, percent=False)
mae_test_10 = get_loss_vs_iterations(clf_10, features_testing, truth_testing, percent=False)
np.save(saving_path + "mae_train_depth_10.npy", mae_train_10)
np.save(saving_path + "mae_test_depth_10.npy", mae_test_10)

