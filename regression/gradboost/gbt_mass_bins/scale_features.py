import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


def get_loss_vs_iterations(clf, features, truth, percent=False):
    mae = np.zeros(len(clf.estimators_), )
    for i, y_pred in enumerate(clf.staged_predict(features)):
        if percent is True:
            mae[i] = np.mean(np.abs((truth - y_pred) / truth)) * 100
        else:
            mae[i] = mean_absolute_error(truth, y_pred)
    return mae


saving_path = "/share/data2/lls/regression/gradboost/halos_range_115_125/scaled_features/"
halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
traj = np.load("/share/data2/lls/features_w_periodicity_fix/ics_density_contrasts.npy")

training_ids = np.load("/share/data2/lls/regression/gradboost/randomly_sampled_training/ic_traj/nest_2000_lr006/"
                       "training_ids.npy")
testing_ids = np.load("/share/data2/lls/regression/gradboost/randomly_sampled_training/ic_traj/nest_2000_lr006/"
                       "testing_ids.npy")

scaler = StandardScaler()
scaler.fit(traj[training_ids])
t = scaler.transform(traj[training_ids])

ids_between_115_125 = np.where((np.log10(halo_mass[training_ids]) > 11.5) &
                               (np.log10(halo_mass[training_ids]) <= 12.5))[0]

### ignore last feature due to sampling of window function in fourier space


features_training = t[ids_between_115_125, :-1]
truth_training = np.log10(halo_mass[training_ids[ids_between_115_125]])

# ids_testing_between_115_125 = np.where((np.log10(halo_mass[testing_ids]) > 11.5) &
#                                         (np.log10(halo_mass[testing_ids]) <= 12.5))[0]
#
#
# features_testing = traj[testing_ids[ids_testing_between_115_125], :-1]
# truth_testing = np.log10(halo_mass[testing_ids[ids_testing_between_115_125]])



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

# mae_train_10 = get_loss_vs_iterations(clf_10, features_training, truth_training, percent=False)
# mae_test_10 = get_loss_vs_iterations(clf_10, features_testing, truth_testing, percent=False)
# np.save(saving_path + "mae_train_depth_10.npy", mae_train_10)
# np.save(saving_path + "mae_test_depth_10.npy", mae_test_10)

